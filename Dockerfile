FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y git gcc && rm -rf /var/lib/apt/lists/*
COPY ser.py /app/ser.py
RUN pip install --no-cache-dir pillow numpy
EXPOSE 8320
ENV PORT=8320
CMD ["python", "ser.py", "--host", "0.0.0.0"]
