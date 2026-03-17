import os
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset

# Load data
df = pd.read_csv("data/raw/train.csv")

# Remove orphan label
df = df[df["label"] != "Orphan"]

# Remove rare labels (keep labels with >= 10 samples)
label_counts = df["label"].value_counts()
valid_labels = label_counts[label_counts >= 10].index
df = df[df["label"].isin(valid_labels)].copy()

# Encode labels
le = LabelEncoder()
df["label_id"] = le.fit_transform(df["label"])

train_df, val_df = train_test_split(df, test_size=0.1, stratify=df["label_id"])

# Convert to HF dataset
train_ds = Dataset.from_pandas(train_df)
val_ds = Dataset.from_pandas(val_df)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding=True,
        max_length=256
    )

train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)

train_ds = train_ds.rename_column("label_id", "labels")
val_ds = val_ds.rename_column("label_id", "labels")

train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Model
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(le.classes_)
)

# Training args
training_args = TrainingArguments(
    output_dir="models/bert",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=7,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
)

# Evaluation metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }

# Trainer
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)

trainer.train()

# Save
model.save_pretrained("models/bert/latest")
tokenizer.save_pretrained("models/bert/latest")

import joblib
joblib.dump(le, "models/bert/latest/label_encoder.pkl")

print("✅ BERT model trained and saved")