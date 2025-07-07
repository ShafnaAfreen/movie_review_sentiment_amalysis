# model_utils.py

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import AutoModelForSequenceClassification
from config import CHECKPOINT, NUM_LABELS

def get_model():
    return AutoModelForSequenceClassification.from_pretrained(CHECKPOINT, num_labels=NUM_LABELS)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }
