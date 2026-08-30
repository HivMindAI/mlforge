FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    MLFORGE_WEB_WORKSPACE=/var/lib/mlforge

WORKDIR /app

COPY pyproject.toml README.md LICENSE MANIFEST.in ./
COPY src ./src

RUN python -m pip install ".[web]" \
    && groupadd --system --gid 10001 mlforge \
    && useradd --system --uid 10001 --gid mlforge --home-dir /nonexistent \
        --shell /usr/sbin/nologin mlforge \
    && mkdir -p /var/lib/mlforge \
    && chown -R mlforge:mlforge /var/lib/mlforge

USER mlforge

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health/ready', timeout=3).read()"]

CMD ["python", "-m", "uvicorn", "mlforge.web.app:create_app", "--factory", \
    "--host", "0.0.0.0", "--port", "8000", "--workers", "1", \
    "--timeout-keep-alive", "5"]
