import json
import re
import torch
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

MODEL_PATH = "models/bert/latest"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
le = joblib.load(f"{MODEL_PATH}/label_encoder.pkl")

model.eval()


def clean_text(input_text):
    try:
        data = json.loads(input_text)
        text = data.get("errorMessage", input_text)
    except Exception:
        text = input_text

    text = text.lower()

    # Remove ids
    text = re.sub(r"\b[a-z0-9]{8,}\b", "", text)

    # Remove noise words
    text = re.sub(r"exception message:", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def rule_based_override(text):
    t = text.lower()

    if "payment" in t or "amount deducted" in t:
        return "Payment Issue"

    if "slot" in t or "availability" in t:
        return "Inventory Issue > Current Slot Unavailable"

    if "timeout" in t or "stream was reset" in t:
        return "System Issue"

    return None


def predict(text):
    text = clean_text(text)

    # Fallback to rule-based override when applicable
    override_label = rule_based_override(text)
    if override_label is not None:
        if override_label in le.classes_:
            idx = list(le.classes_).index(override_label)
            probs = [0.0] * len(le.classes_)
            probs[idx] = 1.0
            return {
                "label": override_label,
                "confidence": 1.0,
                "all_probs": probs
            }
        # Label not in training set, fall through to model

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=1).numpy()[0]

    top_idx = probs.argmax()
    return {
        "label": le.inverse_transform([top_idx])[0],
        "confidence": float(probs[top_idx]),
        "all_probs": probs.tolist()
    }