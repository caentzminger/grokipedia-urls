fmt:
    uv run --dev ruff format main.py

typecheck:
    uv run --dev ty check main.py

lint:
    uv run --dev ruff check main.py

test:
    uv run main.py --help

collect:
    uv run main.py

collect-push:
    uv run main.py --push
