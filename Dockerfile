# Delay-plan compiler service — zero third-party Python dependencies.
FROM python:3.11-slim AS build

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Application code, tests and operational scripts.
COPY app/ ./app/
COPY tests/ ./tests/
COPY scripts/ ./scripts/

# Bake the build-artifact manifest (hashes) into the image.  The one-shot
# verify service re-hashes these files and compares against the manifest.
RUN python scripts/make_build_manifest.py

# Runtime stage keeps the image minimal.
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    API_PORT=8080 \
    HOST=0.0.0.0

WORKDIR /app

RUN groupadd --system app && useradd --system --gid app --home /app app

COPY --from=build --chown=app:app /app /app

USER app

EXPOSE 8080

HEALTHCHECK --interval=5s --timeout=3s --start-period=3s --retries=5 \
    CMD python -c "import os,sys,urllib.request as u; sys.exit(0 if u.urlopen('http://127.0.0.1:%s/healthz' % os.environ['API_PORT'], timeout=2).status == 200 else 1)"

CMD ["python", "-m", "app.server"]
