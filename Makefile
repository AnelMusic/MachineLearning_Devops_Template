# Makefile
# /bin/bash supports source
SHELL := /bin/bash

.PHONY: help
help:
	@echo "Commands:"
	@echo "venv   : creates development environment."
	@echo "style  : runs style formatting."	
	@echo "test  : runs style formatting."
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

# Testing
.PHONY: test
test:
	pytest --cov=tests > test_coverage.txt	

# Cleaning
.PHONY: clean
clean: 
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	rm -f .coverage


