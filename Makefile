.PHONY: install download preprocess train evaluate serve test lint clean all

# Default target
all: install download preprocess train evaluate

install:
	pip install -r requirements.txt

download:
	python -m src.data.download

validate:
	python -m src.data.validate

preprocess:
	python -m src.data.preprocess

train:
	python -m src.models.train

evaluate:
	python -m src.models.evaluate

serve:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

serve-prod:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4

predict:
	@echo "Usage: make predict EIN=53-0196605 NAME='American Red Cross'"
	python -m src.cli predict $(EIN) "$(NAME)"

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff format src/ tests/

clean:
	rm -rf data/raw/ data/processed/ models/ reports/ __pycache__ .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

docker-build:
	docker build -t nonprofit-risk-model .

docker-run:
	docker run -p 8000:8000 nonprofit-risk-model
