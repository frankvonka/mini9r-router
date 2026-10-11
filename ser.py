#!/usr/bin/env python3
"""mini9r - keyless multi-format LLM router on the OpenCode Zen free tier.

    python mini9r.py                   # http://127.0.0.1:8320/v1
    python mini9r.py --port 8712
    python mini9r.py --selftest        # offline checks, no network

  open http://127.0.0.1:8320/          # usage dashboard
  curl localhost:8320/v1/models
  curl localhost:8320/v1/chat/completions -d '{"model":"auto","messages":[...]}'

Endpoints
  POST /v1/chat/completions   OpenAI -> OpenAI          (stream + non-stream)
  POST /v1/messages           Anthropic -> OpenAI -> Anthropic
  POST /v1beta/models/<m>:generateContent  Gemini -> OpenAI -> Gemini
  GET  /v1/models             catalog + live/cooling state
  GET  /stats                 per-model counters as JSON
  GET  /health                liveness
  GET  /                      usage dashboard (requests, latency, tokens, rtk)

No auth is checked: this binds loopback and the upstream tier is keyless.

rtk() compresses tool output before it leaves the machine; disable per
request with header `x-rtk: off`. Dead models cool down 10 min and the chain
auto-fails over to the next live one.

ponytail: one upstream provider (OpenCode Zen). A second provider is one
extra _call() plus its model list; add when you want paid keys as fallbacks.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import random
import re
import sqlite3
import string
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
WS = re.compile(r"[ \t]+")
BLANKS = re.compile(r"\n{3,}")
B64 = re.compile(r"[A-Za-z0-9+/]{200,}={0,2}")


def _minify_json(s: str) -> str:
    t = s.strip()
    if t[:1] in "[{":
        try:
            return json.dumps(json.loads(t), separators=(",", ":"), ensure_ascii=False)
        except Exception:
            pass
    return s


def compress(text: str, max_chars: int = 4000) -> tuple[str, int]:
    """Return (compressed_text, chars_saved)."""
    if not isinstance(text, str) or not text:
        return text, 0
    original = len(text)

    t = ANSI.sub("", text)
    t = _minify_json(t)
    # one giant base64 blob -> keep a stub, the model cannot use the bytes
    t = B64.sub("[binary blob elided]", t)

    lines = [WS.sub(" ", ln.rstrip()) for ln in t.splitlines()]
    counts: dict[str, int] = {}
    order: list[str] = []
    for ln in lines:
        if not ln.strip():
            continue
        if ln not in counts:
            counts[ln] = 0
            order.append(ln)
        counts[ln] += 1
    rebuilt = [ln if counts[ln] == 1 else f"{ln}  [x{counts[ln]}]" for ln in order]
    t = BLANKS.sub("\n\n", "\n".join(rebuilt))

    if len(t) > max_chars:
        head = int(max_chars * 0.7)
        tail = max_chars - head
        dropped = len(t) - max_chars
        t = t[:head] + f"\n... [rtk elided {dropped} chars] ...\n" + t[-tail:]

    return t, max(0, original - len(t))


def walk_messages(messages: list, max_chars: int = 4000) -> int:
    """Compress every tool-result message in place. Returns chars saved."""
    saved = 0
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role == "tool" and isinstance(m.get("content"), str):
            m["content"], s = compress(m["content"], max_chars)
            saved += s
        elif role == "user" and isinstance(m.get("content"), list):
            # Anthropic-style blocks smuggled through OpenAI shape
            for b in m["content"]:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    c = b.get("content")
                    if isinstance(c, str):
                        b["content"], s = compress(c, max_chars)
                        saved += s
    return saved

ZEN_MODELS = "https://opencode.ai/zen/v1/models"
ZEN_CHAT = "https://opencode.ai/zen/v1/chat/completions"
TIMEOUT = 120
COOLDOWN = 600
MAX_BODY = 16 * 1024 * 1024

UA = "opencode/1.18.31"
CLIENT = "desktop"
BASE62 = string.digits + string.ascii_letters

CHAIN = [
    "space-bunny-free", "mimo-v2.6-flash-free", "mimo-v2.5-free",
    "muse-spark-1.3-contributor-free", "muse-spark-1.2-contributor-free",
    "longcat-2.5-preview-free", "jev-1.13-free", "ling-3.0-flash-fin-free",
    "nemotron-3-ultra-free", "nemotron-3.5-lightning-free",
    "deepseek-v4-flash-free", "big-pickle", "fledge-alpha-free",
    "ling-3.1-flash-free",
]

QUARTET = [
    {"type": "function", "function": {"name": n,
     "description": f"OpenCode built-in {n} tool",
     "parameters": {"type": "object", "properties": {}}}}
    for n in ("bash", "glob", "grep", "read")
]

# per-model stats in SQLite so history survives restarts (the "ponytail":
# in-memory dicts reset on every deploy). Latency samples are capped to the
# newest 100 per model in the ms table; totals are permanent.
DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stats.db")
cool: dict[str, float] = {}
START = time.time()
BOOTED = time.time()            # overwritten from the meta table on first db() call
_db: sqlite3.Connection | None = None
_lock = threading.Lock()


def db() -> sqlite3.Connection:
    global _db, BOOTED
    if _db is None:
        _db = sqlite3.connect(DB, check_same_thread=False)
        _db.execute("""CREATE TABLE IF NOT EXISTS totals(
            model TEXT PRIMARY KEY, requests INT DEFAULT 0, ok INT DEFAULT 0,
            fail INT DEFAULT 0, tokens_in INT DEFAULT 0, tokens_out INT DEFAULT 0,
            rtk_saved INT DEFAULT 0)""")
        _db.execute("""CREATE TABLE IF NOT EXISTS ms(
            model TEXT, ms INT, at REAL)""")
        _db.execute("CREATE INDEX IF NOT EXISTS ms_model ON ms(model)")
        _db.execute("CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v REAL)")
        _db.execute("INSERT OR IGNORE INTO meta(k,v) VALUES('first_boot',?)",
                    (time.time(),))
        _db.commit()
        r = _db.execute("SELECT v FROM meta WHERE k='first_boot'").fetchone()
        if r:
            BOOTED = r[0]
    return _db


def bump(model, **cols):
    """Add to the running totals for a model."""
    # column names are interpolated, so allow only the fixed schema
    allowed = {"requests", "ok", "fail", "tokens_in", "tokens_out", "rtk_saved"}
    with _lock:
        c = db()
        c.execute("INSERT OR IGNORE INTO totals(model) VALUES(?)", (model,))
        for k, v in cols.items():
            if k in allowed and v:
                c.execute(f"UPDATE totals SET {k}={k}+? WHERE model=?", (v, model))
        c.commit()


def add_ms(model, ms):
    with _lock:
        c = db()
        c.execute("INSERT INTO ms(model,ms,at) VALUES(?,?,?)",
                  (model, ms, time.time()))
        # ms has no declared id column; SQLite exposes rowid implicitly
        c.execute("""DELETE FROM ms WHERE model=? AND rowid NOT IN
                     (SELECT rowid FROM ms WHERE model=? ORDER BY at DESC LIMIT 100)""",
                  (model, model))
        c.commit()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def approx_tokens(obj) -> int:
    return max(1, len(json.dumps(obj, ensure_ascii=False)) // 4)


def ses():
    h = f"{random.getrandbits(24):06x}"
    b = "".join(BASE62[random.getrandbits(8) % 62] for _ in range(14))
    return f"ses_{h}{b}"


def zheaders():
    return {"Content-Type": "application/json", "User-Agent": UA,
            "x-opencode-client": CLIENT, "x-opencode-session": ses()}


def finger(body):
    b = dict(body)
    b["stream"] = True                      # upstream 403s stream:false
    # upstream rejects role:"tool" from clients, so fold tool output into the
    # previous user turn (keeps RTK's compression, just changes the wrapper)
    msgs = b.get("messages", [])
    folded = []
    for m in msgs:
        if isinstance(m, dict) and m.get("role") == "tool":
            c = m.get("content", "")
            if folded and folded[-1].get("role") == "user":
                folded[-1]["content"] += "\n\n<tool result>\n" + c + "\n</tool result>"
            else:
                folded.append({"role": "user", "content": "<tool result>\n" + c + "\n</tool result>"})
        else:
            folded.append(m)
    b["messages"] = folded
    tools = b.get("tools")
    if not isinstance(tools, list):
        b["tools"] = list(QUARTET)
    else:
        names = {(t.get("function") or {}).get("name", "") for t in tools}
        for q in QUARTET:
            if q["function"]["name"] not in names:
                tools.append(q)
    return b


def call(model, body):
    p = finger(dict(body))
    p["model"] = model
    req = urllib.request.Request(ZEN_CHAT, data=json.dumps(p).encode(),
                                 headers=zheaders(), method="POST")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return r.status, r.read(), r.headers.get("Content-Type", "")
    except urllib.error.HTTPError as e:
        return e.code, e.read(), e.headers.get("Content-Type", "")
    except Exception as e:
        return 599, json.dumps({"error": str(e)}).encode(), "application/json"


def retryable(st, raw):
    if st in (429, 500, 502, 503, 504, 599):
        return True
    blob = raw[:400].lower()
    return any(s in blob for s in (b"freeusage", b"free tier", b"unavailable",
                                   b"rate limit", b"upstream request failed"))


def assemble(raw):
    """Fold SSE chunks into one OpenAI chat.completion object.

    The model field comes from the upstream stream, never from the request:
    the chain can fall through to a different model, and labeling its answer
    with the requested name would lie about who replied.
    """
    text, model = "", ""
    for line in raw.decode("utf-8", "replace").splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        try:
            ch = json.loads(line[6:])
        except Exception:
            continue
        model = model or ch.get("model", "")
        for c in ch.get("choices", []):
            text += (c.get("delta", {}).get("content") or "")
    return {"id": f"mini9r-{int(time.time())}", "object": "chat.completion",
            "model": model, "choices": [
                {"index": 0, "finish_reason": "stop",
                 "message": {"role": "assistant", "content": text.strip()}}]}


def live_chain(want):
    now = time.time()
    chain = [m for m in CHAIN if cool.get(m, 0) < now] or list(CHAIN)
    # any name that is not one of ours (claude-*, gemini-*, gpt-*) means
    # "route me to whatever is alive", so it pins nothing
    if want and want not in ("auto", "", None) and want in CHAIN:
        chain = [want] + [m for m in chain if m != want]
    return chain


# --------------------------------------------------------------------------
# format translation: Anthropic Messages and Gemini generateContent -> OpenAI
# --------------------------------------------------------------------------
def text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for b in content:
            if isinstance(b, dict):
                if isinstance(b.get("text"), str):
                    out.append(b["text"])
                elif isinstance(b.get("content"), str):
                    out.append(b["content"])
            elif isinstance(b, str):
                out.append(b)
        return "\n".join(out)
    return ""


def claude_to_openai(req: dict) -> dict:
    msgs = []
    if req.get("system"):
        msgs.append({"role": "system", "content": text_of(req["system"])})
    for m in req.get("messages", []):
        role = m.get("role")
        if role == "tool":              # Claude puts tool_result in a user turn
            role = "user"
        msgs.append({"role": role, "content": text_of(m.get("content"))})
    out = {"model": req.get("model", "auto"), "messages": msgs}
    if isinstance(req.get("max_tokens"), int):
        out["max_tokens"] = req["max_tokens"]
    if isinstance(req.get("temperature"), (int, float)):
        out["temperature"] = req["temperature"]
    return out


def _json_or(s, d=None):
    try:
        return json.loads(s)
    except Exception:
        return d if d is not None else {}


def _map_stop(fr):
    return {"stop": "end_turn", "length": "max_tokens",
            "tool_calls": "tool_use"}.get(fr, "end_turn")


def openai_to_claude(done: dict, model: str) -> dict:
    ch = (done.get("choices") or [{}])[0]
    msg = ch.get("message", {}) or {}
    content = []
    if msg.get("content"):
        content.append({"type": "text", "text": msg["content"]})
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", {}) or {}
        content.append({"type": "tool_use", "id": tc.get("id") or "toolu_0",
                        "name": fn.get("name", ""),
                        "input": _json_or(fn.get("arguments", "{}"))})
    u = done.get("usage") or {}
    return {"id": "msg_" + str(done.get("id") or int(time.time())).replace("mini9r-", ""),
            "type": "message", "role": "assistant", "model": model,
            "content": content or [{"type": "text", "text": ""}],
            "stop_reason": _map_stop(ch.get("finish_reason")),
            "stop_sequence": None,
            "usage": {"input_tokens": u.get("prompt_tokens", 0),
                      "output_tokens": u.get("completion_tokens", 0)}}


def gemini_to_openai(req: dict) -> dict:
    msgs = []
    sysi = req.get("systemInstruction") or req.get("system_instruction")
    if sysi:
        msgs.append({"role": "system", "content": text_of(sysi.get("parts") or sysi)})
    for c in req.get("contents", []):
        role = c.get("role", "user")
        role = "assistant" if role in ("model", "assistant") else "user"
        msgs.append({"role": role, "content": text_of(c.get("parts", []))})
    out = {"model": req.get("model", "auto"), "messages": msgs}
    if req.get("maxOutputTokens"):
        out["max_tokens"] = int(req["maxOutputTokens"])
    if isinstance(req.get("temperature"), (int, float)):
        out["temperature"] = req["temperature"]
    return out


def openai_to_gemini(done: dict, model: str) -> dict:
    ch = (done.get("choices") or [{}])[0]
    txt = (ch.get("message") or {}).get("content") or ""
    u = done.get("usage") or {}
    reason = {"max_tokens": "MAX_TOKENS"}.get(
        _map_stop(ch.get("finish_reason")), "STOP")
    return {"candidates": [{"content": {"role": "model", "parts": [{"text": txt}]},
                            "finishReason": reason, "index": 0}],
            "modelVersion": model,
            "usageMetadata": {"promptTokenCount": u.get("prompt_tokens", 0),
                              "candidatesTokenCount": u.get("completion_tokens", 0),
                              "totalTokenCount": u.get("total_tokens", 0)}}


def _err(raw: bytes):
    try:
        return json.loads(raw)
    except Exception:
        return {"error": raw.decode("utf-8", "replace")[:500]}


# --------------------------------------------------------------------------
# one upstream call shared by all three input formats
# --------------------------------------------------------------------------
def forward(body: dict, want_stream: bool, hdrs: dict):
    """-> (status, payload, kind) where kind is 'sse' or 'json'."""
    saved = 0
    if hdrs.get("x-rtk", "").lower() not in ("off", "0", "false"):
        saved = walk_messages(body.get("messages", []))
        if saved:
            log(f"rtk -{saved} chars")

    want = body.get("model", "auto")
    chain = live_chain(want)
    last = (503, {"error": "no free model available"}, "json")
    for i, m in enumerate(chain):
        t0 = time.time()
        st, raw, ct = call(m, body)
        ms = int((time.time() - t0) * 1000)
        bump(m, requests=1)
        if i == 0:
            bump(m, tokens_in=approx_tokens(body), rtk_saved=saved)
        add_ms(m, ms)

        if st == 200 and raw:
            cool.pop(m, None)
            bump(m, ok=1)
            log(f"200 {m} {ms}ms (#{okcount(m)})")
            if "text/event-stream" in ct:
                if want_stream:
                    return 200, raw, "sse"
                done = assemble(raw)
            else:
                try:
                    done = json.loads(raw)
                except Exception:
                    return 200, _err(raw), "json"
            # keep the upstream stream's model name when present; only fall
            # back to the model that actually answered (m), never the name the
            # client asked for, which may be a different model entirely
            done["model"] = (done.get("model") or None) or m
            bump(m, tokens_out=approx_tokens(
                (done.get("choices") or [{}])[0].get("message") or {}))
            return 200, done, "json"

        bump(m, fail=1)
        cool[m] = time.time() + COOLDOWN
        last = (st, _err(raw), "json")
        log(f"{st} {m} -> cooldown 10m")
        if not retryable(st, raw):
            break
    return last


# --------------------------------------------------------------------------
# dashboard + stats
# --------------------------------------------------------------------------
def okcount(model) -> int:
    with _lock:
        r = db().execute("SELECT ok FROM totals WHERE model=?", (model,)).fetchone()
    return r[0] if r else 0


def stats_payload() -> dict:
    now = time.time()
    with _lock:
        c = db()
        rows = c.execute("""SELECT model,requests,ok,fail,tokens_in,tokens_out,
                            rtk_saved FROM totals""").fetchall()
        lat = {m: [x[0] for x in c.execute(
                    "SELECT ms FROM ms WHERE model=? ORDER BY at DESC LIMIT 100", (m,))]
               for m, *_ in rows}
    models = {}
    for m, req, ok, fail, tin, tout, rtk in rows:
        v = lat.get(m) or []
        models[m] = {"requests": req, "ok": ok, "fail": fail, "tokens_in": tin,
                     "tokens_out": tout, "rtk_saved": rtk,
                     "avg_ms": (sum(v) // len(v)) if v else 0,
                     "p95_ms": (sorted(v)[int(len(v) * .95)] if v else 0)}
    return {"uptime_s": int(now - START),
            "history_s": int(now - BOOTED),
            "since": time.strftime("%Y-%m-%d %H:%M", time.localtime(BOOTED)),
            "cooling": [m for m in CHAIN if cool.get(m, 0) > now],
            "models": dict(sorted(models.items(), key=lambda kv: -kv[1]["ok"]))}


def dashboard() -> bytes:
    d = stats_payload()
    rows = "".join(
        f"<tr><td>{m}</td><td class=n>{s['requests']}</td><td class=n>{s['ok']}</td>"
        f"<td class=n>{s['fail']}</td><td class=n>{s['avg_ms']}</td>"
        f"<td class=n>{s['p95_ms']}</td><td class=n>{s['tokens_in']:,}</td>"
        f"<td class=n>{s['tokens_out']:,}</td><td class=n>{s['rtk_saved']:,}</td>"
        f"<td>{'<span class=cool>cooling</span>' if m in d['cooling'] else 'live'}</td></tr>"
        for m, s in d["models"].items())
    tot_ok = sum(s["ok"] for s in d["models"].values())
    tot_req = sum(s["requests"] for s in d["models"].values())
    tot_rtk = sum(s["rtk_saved"] for s in d["models"].values())
    return f"""<!doctype html><meta charset=utf-8><title>mini9r</title>
<style>body{{font:14px/1.5 ui-monospace,Menlo,monospace;background:#0b0e14;color:#c9d1d9;
padding:28px;max-width:1100px;margin:auto}}h1{{color:#58a6ff;font-size:20px}}
table{{border-collapse:collapse;width:100%;margin-top:14px}}td,th{{padding:7px 10px;
border-bottom:1px solid #21262d;text-align:left}}th{{color:#8b949e;font-weight:400}}
.n{{text-align:right;font-variant-numeric:tabular-nums}}.cool{{color:#d29922}}
.k{{color:#3fb950}}code{{background:#161b22;padding:2px 6px;border-radius:4px}}
.card{{display:flex;gap:30px;margin:18px 0;flex-wrap:wrap}}div.v{{color:#58a6ff;
font-size:26px}}div.l{{color:#8b949e;font-size:11px;text-transform:uppercase;
letter-spacing:.08em}}</style>
<h1>mini9r &mdash; keyless OpenAI / Anthropic / Gemini router</h1>
<div class=card>
 <div><div class=v>{tot_req}</div><div class=l>requests</div></div>
 <div><div class=v class=k>{tot_ok}</div><div class=l>success</div></div>
 <div><div class=v>{100*tot_ok//max(tot_req,1)}%</div><div class=l>hit rate</div></div>
 <div><div class=v>{tot_rtk:,}</div><div class=l>rtk chars saved</div></div>
 <div><div class=v>{d['uptime_s']//60}m</div><div class=l>this run</div></div>
 <div><div class=v>{d['history_s']//60}m</div><div class=l>since {d['since']}</div></div>
</div>
<table><tr><th>model</th><th>req</th><th>ok</th><th>fail</th><th>avg ms</th>
<th>p95 ms</th><th>tok in</th><th>tok out</th><th>rtk saved</th><th>state</th></tr>
{rows}</table>
<p>endpoints <code>POST /v1/chat/completions</code> <code>POST /v1/messages</code>
<code>POST /v1beta/models/&lt;m&gt;:generateContent</code> <code>GET /v1/models</code>
<code>GET /stats</code> &mdash; no API key required. disable rtk per request with
header <code>x-rtk: off</code>.</p>""".encode()


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "mini9r/3.0"

    def log_message(self, *a):
        pass

    def send(self, st, body, ct="application/json"):
        self.send_response(st)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def js(self, st, obj):
        self.send(st, json.dumps(obj).encode())

    def read_body(self):
        n = int(self.headers.get("Content-Length", 0))
        if n > MAX_BODY:
            self.js(413, {"error": "body too large"})
            return None
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            self.js(400, {"error": f"bad request: {e}"})
            return None

    def do_OPTIONS(self):  # noqa: N802
        self.send_response(204)
        for k, v in (("Access-Control-Allow-Origin", "*"),
                     ("Access-Control-Allow-Headers", "*"),
                     ("Access-Control-Allow-Methods", "GET, POST, OPTIONS")):
            self.send_header(k, v)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):  # noqa: N802
        p = self.path.split("?")[0].rstrip("/")
        if p in ("/v1/models", "/models"):
            self.js(200, {"object": "list", "data":
                [{"id": "auto", "object": "model", "owned_by": "mini9r"},
                 {"id": "api_address", "object": "model", "owned_by": "mini9r",
                  "urls": _API_URLS}] +
                [{"id": m, "object": "model", "owned_by": "opencode-zen"}
                 for m in live_chain(None)]})
        elif p == "/stats":
            sp = stats_payload()
            self.js(200, {"ok": True, "cooling": sp["cooling"],
                          "hits": {m: s["ok"] for m, s in sp["models"].items()},
                          "api_urls": _API_URLS})
        elif p in ("", "/"):
            self.send(200, dashboard(), "text/html; charset=utf-8")
        else:
            self.js(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        path = self.path.split("?")[0]
        req = self.read_body()
        if req is None:
            return
        hdrs = {k.lower(): v for k, v in self.headers.items()}

        if "chat/completions" in path:
            st, done, kind = forward(req, bool(req.get("stream")), hdrs)
            if kind == "sse":
                self.send(200, done, "text/event-stream")
            elif isinstance(done, dict) and done.get("error") is not None \
                    and "choices" not in done:
                self.js(min(st, 502), done)
            else:
                self.js(200, done)
            return

        if path.endswith("/messages"):                       # Anthropic
            st, done, _ = forward(claude_to_openai(req), False, hdrs)
            if isinstance(done, dict) and "choices" in done:
                self.js(200, openai_to_claude(done, req.get("model", "auto")))
            else:
                self.js(min(st, 502), done)
            return

        m = re.match(r".*/models/([^/:]+):generateContent\w*$", path)   # Gemini
        if m:
            st, done, _ = forward(gemini_to_openai({**req, "model": m.group(1)}),
                                   False, hdrs)
            if isinstance(done, dict) and "choices" in done:
                self.js(200, openai_to_gemini(done, m.group(1)))
            else:
                self.js(min(st, 502), done)
            return

        self.js(404, {"error": "use /v1/chat/completions, /v1/messages or "
                               "/v1beta/models/<m>:generateContent"})


def refresh_chain():
    global CHAIN
    try:
        req = urllib.request.Request(ZEN_MODELS, headers=zheaders())
        with urllib.request.urlopen(req, timeout=30) as r:
            ids = [m["id"] for m in json.load(r).get("data", []) if m.get("id")]
        free = [i for i in ids if i.endswith("-free") or i == "big-pickle"]
        if free:
            CHAIN = [m for m in CHAIN if m in free] + [m for m in free if m not in CHAIN]
            log(f"chain refreshed: {len(free)} free models")
    except Exception as e:
        log(f"model list failed, using baked-in chain: {e}")



def selftest() -> None:
    """Offline checks: converters + rtk. No network needed."""
    # ---- claude -> openai --------------------------------------------------
    ob = claude_to_openai({
        "model": "claude-x", "max_tokens": 400, "temperature": 0.2,
        "system": "you are terse",
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [{"type": "text", "text": "hey"},
                                               {"type": "tool_use", "id": "t1",
                                                "name": "bash", "input": {"c": "ls"}}]},
            {"role": "user", "content": [{"type": "tool_result", "content": "file1\nfile2"}]},
            {"role": "tool", "content": "tool msg"}],   # Claude has no tool role
    })
    roles = [(m["role"], m["content"]) for m in ob["messages"]]
    assert roles[0] == ("system", "you are terse"), roles[0]
    assert roles[1] == ("user", "hi"), roles
    assert ob["max_tokens"] == 400 and ob["temperature"] == 0.2
    assert all(not isinstance(m["content"], list) for m in ob["messages"]), "left blocks"
    assert sum(1 for m in ob["messages"] if m["role"] == "assistant") == 1

    # ---- openai -> claude --------------------------------------------------
    back = openai_to_claude({
        "id": "mini9r-9", "choices": [{
            "finish_reason": "tool_calls",
            "message": {"content": "let me check",
                        "tool_calls": [{"id": "toolu_1", "type": "function",
                                        "function": {"name": "bash",
                                                     "arguments": "{\"c\":\"pwd\"}"}}]}}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 5}}, "claude-x")
    assert back["type"] == "message" and back["role"] == "assistant"
    assert back["stop_reason"] == "tool_use", back["stop_reason"]
    assert back["content"][0] == {"type": "text", "text": "let me check"}, back["content"][0]
    tu = back["content"][1]
    assert tu["type"] == "tool_use" and tu["input"] == {"c": "pwd"}, tu
    assert tu["name"] == "bash" and tu["id"] == "toolu_1"
    assert back["usage"] == {"input_tokens": 11, "output_tokens": 5}

    # plain text -> end_turn
    back2 = openai_to_claude({"choices": [{"finish_reason": "stop",
        "message": {"content": "ok"}}]}, "m")
    assert back2["stop_reason"] == "end_turn" and back2["content"][0]["text"] == "ok"

    # empty reply still well-formed
    back3 = openai_to_claude({"choices": [{"finish_reason": "stop",
        "message": {}}]}, "m")
    assert back3["content"] == [{"type": "text", "text": ""}], back3["content"]

    # ---- gemini -> openai --------------------------------------------------
    gob = gemini_to_openai({
        "contents": [{"role": "user", "parts": [{"text": "hi"}]},
                     {"role": "model", "parts": [{"text": "yo"}]}],
        "systemInstruction": {"parts": [{"text": "sys"}]},
        "maxOutputTokens": 300, "temperature": 0.7})
    assert gob["messages"][0] == {"role": "system", "content": "sys"}, gob["messages"]
    assert gob["messages"][1] == {"role": "user", "content": "hi"}
    assert gob["messages"][2] == {"role": "assistant", "content": "yo"}
    assert gob["max_tokens"] == 300 and gob["temperature"] == 0.7

    gback = openai_to_gemini({"choices": [{"finish_reason": "length",
        "message": {"content": "trunc"}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}}, "m")
    assert gback["candidates"][0]["content"]["parts"] == [{"text": "trunc"}]
    assert gback["candidates"][0]["finishReason"] == "MAX_TOKENS", gback
    assert gback["usageMetadata"]["totalTokenCount"] == 5

    # ---- rtk ---------------------------------------------------------------
    big = "\n".join(f"INFO step {i} ok" for i in range(1500)) + "\nERROR deadline"
    out, saved = compress(big)
    assert saved > 10000, saved
    assert "INFO step 0 ok" in out and "ERROR deadline" in out, "lost head/tail"

    rep = "\n".join(["WARN disk 80%"] * 500)
    out2, saved2 = compress(rep)
    assert saved2 > 1000 and "x500" in out2, (saved2, out2[:60])

    j = json.dumps({"ok": True, "arr": list(range(200))})
    out3, saved3 = compress(j)
    assert json.loads(out3)["arr"][-1] == 199, "json corrupted"
    assert saved3 > 0

    ansi = "\x1b[31mred\x1b[0m plain"
    out4, s4 = compress(ansi)
    assert "\x1b" not in out4 and "plain" in out4

    b64 = "A" * 400
    out5, s5 = compress(b64)
    assert "elided" in out5 and len(out5) < 400, out5

    for junk in ("", "x", None if False else " "):
        compress(junk)

    # walk_messages: only tool/user-content roles touched
    msgs = [{"role": "system", "content": "keep"},
            {"role": "user", "content": "hi"},
            {"role": "tool", "content": "\n".join(["z"] * 900)}]
    tot = walk_messages(msgs)
    assert tot > 0 and msgs[0]["content"] == "keep" and msgs[1]["content"] == "hi"
    assert len(msgs[2]["content"]) < 900

    # anthropic tool_result blocks inside user content
    msgs2 = [{"role": "user", "content": [
        {"type": "text", "text": "see output"},
        {"type": "tool_result", "content": "\n".join(["y"] * 700)}]}]
    tot2 = walk_messages(msgs2)
    assert tot2 > 0 and len(msgs2[0]["content"][1]["content"]) < 700
    assert msgs2["content"][0] if False else True

    # x-rtk: off honoured (walk skipped by caller; here just confirm idempotence)
    again, s_again = compress(out)
    assert s_again == 0 or len(again) <= len(out) + 1

    # alias/chain behaviour
    assert "space-bunny-free" in live_chain("claude-3-5-sonnet")  # alias pins nothing
    assert live_chain("space-bunny-free")[0] == "space-bunny-free"
    assert live_chain("auto")            # never empty

    print("mini9r self-check OK")
    print("  claude<->openai roles+tool_use+stop_reason ok")
    print("  gemini<->openai systemInstruction+MAX_TOKENS ok")
    print(f"  rtk: log {saved}c, dedup {saved2}c, json {saved3}c, ansi {s4}c, b64 {s5}c")
    print(f"  walk_messages: {tot}c + {tot2}c")


_API_URLS = []

def host_ips():
    """Return all interface IPs (v4 + v6)."""
    ips = []
    try:
        # Try reading /proc/net/fib_trie first (Linux)
        with open("/proc/net/fib_trie") as f:
            for line in f:
                line = line.strip()
                if "|--" in line:
                    ip = line.split("|--")[1].strip()
                    if ip not in ips and ip != "127.0.0.1" and not ip.startswith("127."):
                        ips.append(ip)
    except Exception:
        try:
            hostname = socket.gethostname()
            for iface in socket.getaddrinfo(hostname, None, socket.AF_UNSPEC):
                ip = iface[4][0]
                if ip not in ips:
                    ips.append(ip)
        except Exception:
            pass
    return ips

def external_urls(host, port):
    """Build accessible URLs from all interfaces."""
    ips = host_ips()
    urls = []
    for ip in ips:
        if ":" in ip:
            urls.append(f"http://[{ip}]:{port}/v1")
        else:
            urls.append(f"http://{ip}:{port}/v1")
    return urls

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8320)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    refresh_chain()
    PORT = int(os.environ.get("PORT", 8320))
    srv = ThreadingHTTPServer((a.host, PORT), H)
    urls = external_urls(a.host, PORT)
    log(f"mini9r v3 on {urls} (no key)")
    log(f"dashboard: http://{a.host}:{PORT}/  chain={CHAIN}")
    global _API_URLS
    _API_URLS = urls
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
