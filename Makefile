# Makefile for Airborne Track Tagger

.PHONY: help install build test clean run-gui run-simulator train docs

help:
	@echo "Available targets:"
	@echo "  install       - Install dependencies and build C++ extensions"
	@echo "  build         - Build C++ extensions only"
	@echo "  test          - Run tests"
	@echo "  test-cov      - Run tests with coverage"
	@echo "  clean         - Remove build artifacts"
	@echo "  run-gui       - Launch GUI application"
	@echo "  run-simulator - Run simulator with default settings"
	@echo "  train         - Train ML models on synthetic data"
	@echo "  docs          - View documentation"
	@echo "  format        - Format code with black"
	@echo "  lint          - Lint code with flake8"

install:
	pip install -r requirements.txt
	python setup.py build_ext --inplace
	pip install -e .

build:
	python setup.py build_ext --inplace

test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=. --cov-report=html --cov-report=term
	@echo "Coverage report: htmlcov/index.html"

clean:
	rm -rf build/ dist/ *.egg-info
	rm -rf htmlcov/ .coverage .pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.so" -delete

run-gui:
	python -m gui.main

run-simulator:
	python -m simulator.main --num-tracks 10 --duration 30 --format both --output-dir ./data/simulated

train:
	python scripts/train_models.py --num-tracks 100 --models rf xgb

docs:
	@echo "Documentation files:"
	@echo "  - README.md"
	@echo "  - docs/USER_GUIDE.md"
	@echo "  - docs/API.md"
	@cat README.md | head -50

format:
	@command -v black >/dev/null 2>&1 || { echo "black not installed. Run: pip install black"; exit 1; }
	black .

lint:
	@command -v flake8 >/dev/null 2>&1 || { echo "flake8 not installed. Run: pip install flake8"; exit 1; }
	flake8 --max-line-length=100 --exclude=venv,build

demo:
	@echo "Running demo..."
	python -m simulator.main --num-tracks 5 --duration 20 --format both --output-dir ./data/demo
	@echo "\nSimulated data created in ./data/demo"
	@echo "Now launching GUI..."
	python -m gui.main
