install-req:
	pip install --upgrade pip
	pip install -r requirements-dev.txt


install-pc:
	pre-commit install


pc:
	pre-commit run --all-files


test:
	pytest tests -v


create-package:
	python setup.py sdist bdist_wheel


install-twine:
	pip install twine


twine-upload:
	twine upload dist/*
