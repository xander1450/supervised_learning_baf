import os

import pandas as pd

train = pd.read_csv("data/raw/train.csv")

if os.path.exists("data/feedback/labeled.csv"):
    feedback = pd.read_csv("data/feedback/labeled.csv")
    if len(feedback) > 0:
        updated = pd.concat([train, feedback]).drop_duplicates()
    else:
        updated = train
else:
    updated = train

updated.to_csv("data/raw/train.csv", index=False)

print("✅ Training data updated")
