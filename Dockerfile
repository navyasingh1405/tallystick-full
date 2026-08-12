# Tallystick service container.
#
# Two things drive the shape of this file:
#
# 1. **Playwright's Chromium is 428 MB and needs system libraries.** Building from
#    Microsoft's Playwright image gets both, already matched to the driver version.
#    Installing Chromium onto a slim base means chasing a list of shared objects that
#    changes with every Playwright release.
#
# 2. **The ledger needs a real filesystem that survives restarts.** `/data` is a
#    mounted volume. Nothing that must persist is written into the image layer, which
#    is discarded on every deploy.

FROM mcr.microsoft.com/playwright/python:v1.47.0-jammy

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TALLY_ROOT=/app

WORKDIR /app

# Dependencies first, in their own layer: they change far less often than the source,
# so a code edit does not reinstall 800 MB of scientific Python.
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir .

# The Playwright base image ships browsers for the `pwuser` account; make sure the
# runtime user can reach the one we actually print PDFs with.
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

COPY config/ ./config/
COPY web/ ./web/
COPY docs/ ./docs/

# Model, metrics, fairness report, alternative search, drift. These are build
# products, not source, and the image is the only place they can live on a host with
# an ephemeral filesystem — so they are baked in deliberately. Regenerating them
# means a rebuild, which is the correct coupling: the validation pack must describe
# the model that is actually serving.
COPY models/ ./models/
COPY artifacts/ ./artifacts/

# Persisted across deploys, mounted as a volume. The decision ledger and the account
# store both live here; everything else in the image is disposable.
RUN mkdir -p /data && ln -sfn /data /app/ledger

ENV TALLY_HOST=0.0.0.0 \
    TALLY_PORT=8080 \
    TALLY_COOKIE_SECURE=true

EXPOSE 8080

# One worker, always. The ledger is a single-writer SQLite database and the hash
# chain assumes a single appender: two workers would both read the same head and
# write rows claiming the same predecessor. This is a correctness constraint, not a
# performance tuning choice — do not raise it to "use the cores".
CMD ["uvicorn", "tally.serving.app:app", \
     "--host", "0.0.0.0", "--port", "8080", "--workers", "1", \
     "--log-level", "warning", "--timeout-keep-alive", "65"]
