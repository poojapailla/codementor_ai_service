# syntax=docker/dockerfile:1
#
# CodeMentor AI - FastAPI AI service (GroqCloud backed)
# Build context: repository root (this directory)
#   docker build -t codementor-ai-service .
#
# Render injects PORT at runtime; default 8000 is used for local runs.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

# Install dependencies first so this layer is cached across code changes.
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Application code only (tests are not needed in the runtime image).
COPY main.py ./

# Run as a non-root user.
RUN useradd --create-home --uid 1001 app \
    && chown -R app:app /app
USER app

EXPOSE 8000

# Bind to 0.0.0.0 and honour Render's assigned PORT.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
