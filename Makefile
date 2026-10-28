PY ?= python
.PHONY: install test eval up lint
install: ; $(PY) -m pip install -e ".[dev]"
test: ; $(PY) -m pytest -q
eval: ; $(PY) evals/run_eval.py --check
lint: ; ruff check src tests evals
up: ; docker compose up --build
