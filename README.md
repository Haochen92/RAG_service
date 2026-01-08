# RAG Service (pgvector + FastAPI)

Toy RAG project with:
- Python backend managed by Poetry
- Postgres + pgvector via Docker Compose
- Jupyter notebooks for experiments
- (Later) Mantine UI frontend

## Prereqs
- Python 3.11
- Poetry
- Docker Desktop (or Docker Engine + Compose)

## Setup (Poetry + VS Code)
1) Create the in-project virtualenv:
```bash
poetry config virtualenvs.in-project true
poetry install --with dev,notebook
poetry run pre-commit install
```

2) VS Code:
- Open the repo folder
- Select interpreter: `.venv/bin/python` (the repo includes `.vscode/settings.json`)

## Start infra (pgvector)
```bash
cp .env.example .env
docker compose up -d db
```

Database initialization runs from `infra/postgres/init.sql` on first boot.

Optional tools (pgAdmin + Redis):
```bash
docker compose --profile tools up -d pgadmin redis
```

Note: the demo schema uses `vector(768)` in `infra/postgres/init.sql`. If you change the embedding dimension,
recreate the DB volume (`docker compose down -v`) or add a migration later.

## Run the API
```bash
poetry run uvicorn rag_service.main:app --reload --port 8000
curl -s http://localhost:8000/health
```

## Notebooks
```bash
poetry run python -m ipykernel install --user --name rag-service
poetry run jupyter lab
```

## Frontend (later)
Recommended structure: `frontend/` (React + Mantine + Vite).

## Handy commands
```bash
make infra-up
make api
```
