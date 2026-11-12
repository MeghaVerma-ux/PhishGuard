FROM python:3.11-slim

WORKDIR /app

COPY requirements-serve.txt .

RUN pip install --no-cache-dir -r requirements-serve.txt

COPY app.py .
COPY model/ model/
COPY data/sample_input.json data/sample_input.json
COPY results/ results/
COPY static/ static/

EXPOSE 10000

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000}"]