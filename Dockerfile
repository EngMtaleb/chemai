FROM python:3.11-slim AS builder
WORKDIR /build
COPY pyproject.toml ./
COPY chemai/ ./chemai/
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir --prefix=/install .

FROM python:3.11-slim
RUN useradd --create-home --uid 1000 appuser
WORKDIR /app
COPY --from=builder /install /usr/local
COPY projects/ ./projects/
COPY models/ ./models/
RUN mkdir -p /app/data && chown -R appuser:appuser /app
USER appuser
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 CHEMAI_DATA_DIR=/app/data CHEMAI_MODEL_PATH=/app/models/soft_sensor.pkl
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,json,sys; d=json.loads(urllib.request.urlopen('http://localhost:8000/health').read()); sys.exit(0 if d['model_fitted'] else 1)"
CMD ["uvicorn", "chemai.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
