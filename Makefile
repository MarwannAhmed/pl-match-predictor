.PHONY: all install test pipeline collect validate engineer preprocess visualize select train

all: install test pipeline

install:
	poetry install --no-root

test:
	poetry run pytest

pipeline: collect validate engineer preprocess visualize select train

collect:
	poetry run python src/data/collect/collect.py

validate:
	poetry run python src/data/validate/validate.py

engineer:
	poetry run python src/features/engineer/engineer.py

preprocess:
	poetry run python src/features/preprocess/preprocess.py

visualize:
	poetry run python src/features/visualize/visualize.py

select:
	poetry run python src/features/select/_select.py

train:
	poetry run python src/models/train.py

coverage:
	poetry run pytest --cov=src --cov-report=xml

mlflow-ui:
	poetry run mlflow ui