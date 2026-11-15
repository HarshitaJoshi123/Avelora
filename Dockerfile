FROM python:3.12-slim

WORKDIR /app

# uv gives you the `uvx` command your AviationStack MCP server needs
RUN pip install --no-cache-dir uv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000}"]