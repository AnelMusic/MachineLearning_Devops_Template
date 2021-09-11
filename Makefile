# Makefile
# /bin/bash supports source
SHELL := /bin/bash

.PHONY: help
help:
	@echo "Commands:"
	@echo "venv   : creates development environment."
	@echo "style  : runs style formatting."
	@echo "pytest : runs pytest."
	@echo "dvc    : pushes versioned artifacts to blob storage."	
	@echo "api : Launch restapi on ASGI server."
	@echo "clean  : cleans all unecessary files."


# Environment
.ONESHELL:
venv:
	python3 -m venv venv
	source venv/bin/activate && \
	python -m pip install --upgrade pip setuptools wheel && \
	python -m pip install -e ".[dev]"


# Styling
.PHONY: style
style:
	black . --exclude ./venv
	flake8 # venv excluded in .flake8 file
	isort .

# Py Testing
.PHONY: pytest
pytest:
	pytest --cov=tests > test_coverage.txt  && \
	pytest -v

# DVC
.PHONY: dvc
dvc:
	dvc add data/prediction_data.csv
	dvc add data/titanic.csv
	dvc add data/titanic_processed.csv
	dvc add data/titanic_test_processed.csv
	dvc add data/titanic_train_processed.csv
	dvc push

# Clean
.PHONY: clean
clean: style
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf	

# Clean
.PHONY: api
api:
	uvicorn app.api:app \
	--host 0.0.0.0 \
	--port 5000 \
	--reload \
	--reload-dir titanic_classification \
	--reload-dir app
	




















