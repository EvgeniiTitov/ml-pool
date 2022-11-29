install-req:
	pip install --upgrade pip
	pip install -r requirements-dev.txt


install-pc:
	pre-commit install


pc:
	pre-commit run --all-files


test:
	pytest tests -v