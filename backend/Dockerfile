FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY alembic.ini .
COPY alembic ./alembic
RUN mkdir -p /data /app/data/models

ENV CHECKPOINT_PATH=/data/checkpoints.sqlite
ENV ORG_CLOCK=2026-11-07
ENV EMBEDDING_PROVIDER=local
ENV EMBEDDING_MODEL=minishlab/potion-retrieval-32M
ENV EMBEDDING_DIMENSIONS=512
ENV HF_HOME=/app/data/models

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
