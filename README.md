# Text Classification System (BERT-based)

## Overview

This project is a production-ready text classification system built using DistilBERT. It takes raw text input (e.g., customer issues) and predicts the most relevant category along with a confidence score.

The system is designed to:

* Train on labeled text data
* Serve predictions via API
* Log low-confidence predictions for human review
* Continuously improve via feedback loop and retraining

![API Docs](images/Screenshot%202026-03-17%20at%206.45.05%20PM.png)

![API Example](images/Screenshot%202026-03-17%20at%206.45.29%20PM.png)

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

* `distilbert-base-uncased`

Why DistilBERT?

* Faster and lighter than BERT
* Understands context (not just keywords)
* Often better on small datasets
* Performs significantly better than TF-IDF models

---

### 3. Tokenization

Text is converted into numerical format using the DistilBERT tokenizer:

* Max sequence length: 256
* Truncation enabled
* Dynamic padding (per batch)

---

### 4. Model Training

We fine-tune DistilBERT using:

* Epochs: 7
* Learning rate: 2e-5
* Warmup ratio: 0.1
* Train/Validation split: 90/10

Training is handled via HuggingFace `Trainer` with accuracy and F1 evaluation.

---

### 5. Output Predictions

For each input text, the model returns:

* Predicted label
* Confidence score (probability)
* All class probabilities

When confidence is below 0.5, the label is returned as `"uncertain"` and the text is logged to `data/feedback/unlabeled.csv` for human review.

Example:

```
{
  "label": "Inventory Issue > Current Slot Unavailable",
  "confidence": 0.78,
  "all_probs": [...]
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
python3 -m uvicorn app.api:app --reload
```

2. Open docs:

```
http://127.0.0.1:8000/docs
```

3. Test `/predict`

---

## Feedback Loop and Retraining

The system supports continuous improvement via human feedback:

### 1. Low-confidence logging

Predictions with confidence < 0.5 are logged to `data/feedback/unlabeled.csv` and returned as `"uncertain"`.

### 2. Human labeling

Review `data/feedback/unlabeled.csv` and add correct labels to `data/feedback/labeled.csv`:

```csv
text,label
"payment failed again","Payment Issue"
"slot unavailable","Inventory Issue"
```

### 3. Merge and retrain

Run the retrain script to merge new labels into training data and retrain the model:

```bash
bash scripts/retrain.sh
```

This runs `scripts/merge_feedback.py` (merges `labeled.csv` into `train.csv`) followed by `python3 -m training.train_bert`.

### 4. Optional: scheduled retraining

To retrain daily at 2 AM (Mac/Linux):

```bash
crontab -e
```

Add:

```
0 2 * * * /path/to/project/scripts/retrain.sh
```

---

## Performance Expectations

| Stage              | Confidence | Accuracy |
| ------------------ | ---------- | -------- |
| Initial (TF-IDF)   | ~0.15–0.30 | ~55%     |
| BERT (raw data)    | ~0.30–0.45 | ~60%     |
| Cleaned + balanced | ~0.60–0.80 | ~70–85%  |

---

## Key Improvements Made

* Removed noisy labels (Orphan, rare classes)
* Reduced class imbalance
* Increased text context (max_length = 256)
* Upgraded to DistilBERT (faster, often better on small data)
* Text normalization and rule-based overrides
* Low-confidence logging for human review
* Feedback loop: label → merge → retrain

---

## Summary

We transformed a basic text classification system into a production-ready ML pipeline by:

* Cleaning and structuring data
* Using DistilBERT for context-aware predictions
* Handling class imbalance and noisy labels
* Serving predictions via API
* Logging uncertain predictions for human review
* Supporting continuous improvement via feedback and retraining

This setup is scalable, extensible, and ready for real-world use.
