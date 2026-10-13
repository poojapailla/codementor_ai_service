# CodeMentor AI — AI Service

FastAPI service that powers the AI features of CodeMentor AI (code review,
tutor mode, test case generation) using the GroqCloud hosted LLM API.

## Why this exists as a separate service

- Keeps AI calls off the Spring Boot backend (no blocking JVM threads).
- Works on cloud hosts (Render) without a local GPU or Ollama.

## Configuration (all via environment variables)

| Variable         | Required | Purpose                                          |
|------------------|----------|--------------------------------------------------|
| `GROQ_API_KEY`   | yes      | GroqCloud API key (https://console.groq.com/keys)|
| `GROQ_MODEL`     | no       | Model id; auto-discovers if unset                |
| `GROQ_TIMEOUT`   | no       | Seconds per request (default 300)                |

Secrets are provided via the environment or a local `.env` file. `.env` is
git-ignored; copy `.env.example` → `.env` and fill in your real key.

## Endpoints

- `GET /` and `GET /health` — status checks
- `POST /review` — AI code review
- `POST /generate-tests` — AI test case generation

## Run locally

```powershell
# from this folder, with the venv active (or system Python)
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Tests

```powershell
pytest -q
```

Tests mock the Groq client — no API key is needed.

## Deploy (Render)

- Web Service ← this repository
- Runtime: Python
- Build: `pip install -r requirements.txt`
- Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Set env vars: `GROQ_API_KEY`, optionally `GROQ_MODEL`, `GROQ_TIMEOUT`

## Related repositories

- Frontend: https://github.com/poojapailla/codementor_frontend
- Backend: https://github.com/poojapailla/codementor_backend