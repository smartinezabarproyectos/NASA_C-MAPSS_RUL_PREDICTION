.PHONY: install eda train clean

install:
	pip install -r requirements.txt
	pip install -e .

eda:
	jupyter notebook notebooks/01_eda.ipynb

train:
	python -m src.models.train

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
