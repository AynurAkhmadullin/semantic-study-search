# syntax=docker/dockerfile:1

FROM python:3.13-slim-bookworm AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUTF8=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 tini \
    && rm -rf /var/lib/apt/lists/*

FROM base AS builder

COPY --from=ghcr.io/astral-sh/uv:0.11.29 /uv /usr/local/bin/uv
ENV UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --all-extras --no-dev --no-install-project

COPY README.md ./
COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --all-extras --no-dev --no-editable

FROM base AS runtime

RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home --shell /usr/sbin/nologin app \
    && install -d -o app -g app /data /data/library /data/huggingface /data/huggingface/hub

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY .streamlit/config.toml ./.streamlit/config.toml

ENV PATH="/app/.venv/bin:$PATH" \
    SSS_INDEX_DIR=/data/library \
    SSS_MODEL_CACHE=/data/huggingface/hub \
    HF_HOME=/data/huggingface

USER app
VOLUME ["/data"]
EXPOSE 8000 8501
STOPSIGNAL SIGTERM

HEALTHCHECK --interval=30s --timeout=10s --start-period=15m --retries=3 \
    CMD ["python", "-c", "from urllib.request import ProxyHandler, build_opener; opener = build_opener(ProxyHandler({})); [opener.open(url, timeout=3).close() for url in ('http://127.0.0.1:8000/health', 'http://127.0.0.1:8501/_stcore/health')]"]

ENTRYPOINT ["/usr/bin/tini", "--", "semantic-study-start", "--host", "0.0.0.0"]
