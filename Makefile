.PHONY: install demo train test status clean

install:
	pip install -e ".[dev]"

test:
	python -m unittest discover -s tests -t . -v

demo:
	python scripts/demo_integration.py

train:
	python scripts/train_toy.py

status:
	python -m body_of_secrets.cli status

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf *.egg-info build dist .pytest_cache .mypy_cache .ruff_cache runs
