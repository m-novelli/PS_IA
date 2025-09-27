# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

# Boas práticas de env
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# Dependências do sistema (ajuste conforme seus pacotes Python)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ \
    libpq-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copie primeiro requirements para aproveitar cache
COPY requirements.txt .

#
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Agora o código
COPY . .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

