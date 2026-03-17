# Text Classification System (BERT-based)

## Overview

This project is a production-ready text classification system built using BERT. It takes raw text input (e.g., customer issues) and predicts the most relevant category along with a confidence score.

The system is designed to:

* Train on labeled text data
* Serve predictions via API
* Continuously improve with new data

---

## Problem Statement

We are classifying customer issue text into predefined categories.

Challenges in the dataset:

* Large number of labels (~50+)
* Many labels with very few examples (class imbalance)
* Noisy fallback labels (e.g., "Orphan")

These issues lead to low model confidence and poor generalization.

---

## Solution Approach

### 1. Data Preprocessing

We clean and prepare the dataset before training:

* Remove noisy labels:

  * "Orphan" is removed as it does not represent a meaningful category

* Remove rare classes:

  * Labels with fewer than 10 samples are dropped
  * This reduces noise and improves model learning

* Result:

  * Fewer but stronger labels (~20–25)
  * Better class balance

---

### 2. Model Selection

We use:

* `bert-base-uncased`

Why BERT?

* Understands context (not just keywords)
* Handles real-world language variations
* Performs significantly better than TF-IDF models

---

### 3. Tokenization

Text is converted into numerical format using BERT tokenizer:

* Max sequence length: 256
* Truncation enabled
* Padding applied

Why?

* Allows model to capture more context from longer inputs

---

### 4. Handling Class Imbalance

We apply class weighting during training:

* Rare classes are given higher importance
* Frequent classes are slightly penalized

This is done using:

* `compute_class_weight` from sklearn
* Custom loss function in Trainer

---

### 5. Model Training

We fine-tune BERT using:

* Epochs: 5
* Learning rate: 2e-5
* Train/Validation split: 90/10

Training is handled via HuggingFace `Trainer`.

---

### 6. Output Predictions

For each input text, the model returns:

* Predicted label
* Confidence score (probability)
* Optional: top-3 predictions

Example:

```
{
  "label": "Inventory Issue > Current Slot Unavailable",
  "confidence": 0.78
}
```

---

## API Layer

We use FastAPI to expose the model.

### Endpoints

* `GET /health` → health check
* `POST /predict` → get prediction

### Input format

```
{
  "text": "payment failed but amount deducted"
}
```

---

## Training Flow

1. Place dataset in:
   `data/raw/train.csv`

2. Run training:

```
python3 -m training.train_bert
```

3. Model is saved in:

```
models/bert/latest
```

---

## Inference Flow

1. Start API:

```
uvicorn app.api:app --reload
```

2. Open docs:

```
http://127.0.0.1:8000/docs
```

3. Test `/predict`

---

## Performance Expectations

| Stage              | Confidence | Accuracy |
| ------------------ | ---------- | -------- |
| Initial (TF-IDF)   | ~0.15–0.30 | ~55%     |
| BERT (raw data)    | ~0.30–0.45 | ~60%     |
| Cleaned + balanced | ~0.60–0.80 | ~70–85%  |

---

## Key Improvements Made

* Removed noisy labels
* Reduced class imbalance
* Increased text context (max_length = 256)
* Added class-weighted loss
* Upgraded model from TF-IDF → BERT

---

## Future Improvements

* Continuous learning from new labeled data
* Active learning for uncertain predictions
* Deployment with Docker
* Monitoring and feedback loop

---

## Summary

We transformed a basic text classification system into a production-ready ML pipeline by:

* Cleaning and structuring data
* Using a powerful language model (BERT)
* Handling class imbalance properly
* Serving predictions via API

This setup is scalable, extensible, and ready for real-world use.
