.PHONY: infra-up infra-down infra-reset api notebook fmt lint test

infra-up:
	docker compose up -d db

infra-down:
	docker compose down

infra-reset:
	docker compose down -v

api:
	poetry run uvicorn rag_service.main:app --reload --port 8000

notebook:
	poetry run jupyter lab

fmt:
	poetry run ruff check . --fix
	poetry run black .

lint:
	poetry run ruff check .
	poetry run mypy src

test:
	poetry run pytest -q
