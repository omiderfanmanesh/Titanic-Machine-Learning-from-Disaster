.PHONY: help install test test-unit test-integration lint format type-check quality clean dev-setup
.DEFAULT_GOAL := help

help: ## Display this help message
	@echo "Titanic ML Pipeline - Development Commands"
	@echo "========================================"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install package in development mode
	pip install -e .

dev-setup: ## Set up development environment
	pip install -e ".[dev]"
	pre-commit install

test: ## Run all tests with coverage
	pytest tests/ --cov=titanic_ml --cov-report=html --cov-report=term --cov-fail-under=80

test-unit: ## Run unit tests only
	pytest tests/unit/ -v

test-integration: ## Run integration tests only
	pytest tests/integration/ -v

test-fast: ## Run tests without coverage for faster feedback
	pytest tests/ --no-cov -x

lint: ## Run linting checks
	ruff check src/ tests/
	ruff format --check src/ tests/

format: ## Format code using black and ruff
	black src/ tests/
	ruff format src/ tests/

type-check: ## Run type checking with mypy
	mypy src/titanic_ml/

quality: lint type-check ## Run all quality checks
	@echo "All quality checks passed! ✨"

clean: ## Clean build artifacts and cache
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

# Development workflows
dev-test: ## Quick development test cycle
	@echo "Running quick development tests..."
	pytest tests/unit/test_core.py -v --no-cov

dev-lint: ## Quick lint check for development
	ruff check src/titanic_ml/ --select E,W,F

# Data and training commands
download-data: ## Download Kaggle Titanic data
	titanic-ml download --competition titanic

train-baseline: ## Train baseline logistic regression model
	titanic-ml train --config configs/experiment.yaml --model logistic

train-ensemble: ## Train ensemble of models
	titanic-ml train --config configs/ensemble.yaml

# Docker commands (if using Docker)
docker-build: ## Build Docker image
	docker build -t titanic-ml .

docker-test: ## Run tests in Docker container
	docker run --rm titanic-ml pytest

# Profiling and debugging
profile-training: ## Profile training performance
	python -m cProfile -o profile_training.prof -m titanic_ml.cli train --config configs/experiment.yaml
	@echo "Profile saved to profile_training.prof"

profile-features: ## Profile feature engineering performance
	python -m cProfile -o profile_features.prof -m titanic_ml.cli features --config configs/experiment.yaml
	@echo "Profile saved to profile_features.prof"

# Documentation
docs: ## Generate documentation
	@echo "Documentation generation not implemented yet"

# Release commands
check-release: ## Check if ready for release
	@echo "Checking release readiness..."
	make quality
	make test
	@echo "Ready for release! 🚀"

release-patch: ## Create patch release
	bump2version patch
	git push --tags

release-minor: ## Create minor release
	bump2version minor  
	git push --tags

# Jupyter notebook commands
notebook: ## Start Jupyter notebook
	jupyter notebook notebooks/

lab: ## Start Jupyter lab
	jupyter lab notebooks/

# Environment management
freeze: ## Freeze current environment to requirements
	pip freeze > requirements-frozen.txt

# Kaggle specific commands
submit: ## Create and submit Kaggle submission (requires trained model)
	titanic-ml predict --model-path artifacts/latest/model_logistic.joblib --output submission.csv
	titanic-ml submit --file submission.csv

# Debugging and development utilities
debug-config: ## Debug configuration loading
	python -c "from titanic_ml.core.utils import ConfigManager; cm = ConfigManager(); print(cm.load_config('configs/experiment.yaml'))"

debug-features: ## Debug feature engineering with sample data
	python -c "from titanic_ml.features.build import TitanicFeatureBuilder; from titanic_ml.data.loader import TitanicDataLoader; loader = TitanicDataLoader('data/train.csv', 'data/test.csv'); train, _ = loader.load(); builder = TitanicFeatureBuilder(); X = builder.fit_transform(train); print(f'Features: {X.columns.tolist()}'); print(f'Shape: {X.shape}')"

# Performance monitoring
benchmark: ## Run performance benchmarks
	python scripts/benchmark_pipeline.py

# Security checks
security: ## Run security vulnerability checks
	safety check
	bandit -r src/

# Git hooks and pre-commit
pre-commit-all: ## Run pre-commit on all files
	pre-commit run --all-files

# Help for specific areas
help-testing: ## Show testing help
	@echo "Testing Commands:"
	@echo "  make test         - Full test suite with coverage"
	@echo "  make test-unit    - Fast unit tests only"  
	@echo "  make test-integration - End-to-end integration tests"
	@echo "  make test-fast    - Tests without coverage for speed"

help-quality: ## Show code quality help
	@echo "Code Quality Commands:"
	@echo "  make lint         - Check code style and errors"
	@echo "  make format       - Auto-format code"
	@echo "  make type-check   - Run type checking"
	@echo "  make quality      - Run all quality checks"
