.PHONY: help install format lint test run ingest docker-up docker-down

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install python dependencies
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install black flake8 pytest

format: ## Format code using Black
	black rag/ app/

lint: ## Run Flake8 linter
	flake8 rag/ app/

test: ## Run tests
	pytest tests/ -v

run: ## Run the Streamlit frontend locally
	streamlit run app/main.py

ingest: ## Run document ingestion pipeline
	python rag/pipeline.py ingest

docker-up: ## Start all services via docker-compose
	docker-compose up --build -d

docker-down: ## Stop all docker services
	docker-compose down

clean: ## Remove pycache
	find . -type d -name "__pycache__" -exec rm -rf {} +
