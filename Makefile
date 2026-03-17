PYTHON ?= python
PORT ?= 8000
TEXT ?= please refund my order

bootstrap:
	$(PYTHON) scripts/bootstrap_data.py

train:
	$(PYTHON) -m training.train --train-path data/raw/train.csv

serve:
	uvicorn app.api:app --reload --port $(PORT)

predict:
	$(PYTHON) scripts/predict_cli.py --text "$(TEXT)"

retrain:
	$(PYTHON) -m training.retrain --base-train-path data/raw/train.csv --new-data-path data/raw/new_labeled.csv --promote-data

test:
	pytest -q

docker-build:
	docker build -t text-classifier-starter .

docker-run:
	docker run -p 8000:8000 text-classifier-starter
