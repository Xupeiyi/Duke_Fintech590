install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

lint:
	python -m pylint --disable=R,C febrisk

test:
	python -m pytest -v --reruns 3  --reruns-delay 5 --cov=febrisk tests

all: install lint test