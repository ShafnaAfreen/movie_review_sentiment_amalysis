# Model Report

## Project Overview
This project implements a sentiment analysis model trained on the IMDb movie review dataset. The goal is to classify movie reviews as **positive** or **negative** using a fine-tuned transformer model.

---

## Model Architecture
- **Base Model:** DistilBERT (`distilbert-base-uncased`)
- **Task:** Sequence classification with 2 labels (positive, negative)
- **Framework:** Hugging Face Transformers
- **Tokenizer:** Pretrained DistilBERT tokenizer with padding and truncation to fixed length

---

## Data
- **Dataset:** IMDb dataset from the Hugging Face Datasets library
- **Train Size:** 5000 samples (random subset)
- **Test Size:** 20% split from the train subset
- **Preprocessing:** Tokenization with padding (`max_length=128`), truncation, and label renaming

---

## Training Details
- **Epochs:** 3
- **Batch Size:** 16
- **Learning Rate:** 2e-5
- **Weight Decay:** 0.01
- **Optimizer:** AdamW (default from Trainer)
- **Evaluation Strategy:** Evaluation at the end of each epoch
- **Early Stopping:** Not implemented, but best model loaded at end based on F1 score

---

## Evaluation Metrics

| Metric    | Score   |
|-----------|---------|
| Accuracy  | 90.3%   |
| Precision | 89.3%   |
| Recall    | 90.9%   |
| F1 Score  | 90.15%  |

The model demonstrates strong performance with balanced precision and recall, indicating effective sentiment classification.

---

## Confusion Matrix

|                  | Predicted Negative | Predicted Positive |
|------------------|--------------------|--------------------|
| **Actual Negative** | 459 (True Negatives) | 53 (False Positives) |
| **Actual Positive** | 44 (False Negatives) | 444 (True Positives) |

---

## Insights and Improvements
- The model effectively distinguishes between positive and negative reviews with minimal misclassifications.
- Larger training dataset or additional epochs might improve performance.
- Experimenting with other transformer architectures (e.g., BERT, RoBERTa) could further boost accuracy.
- Incorporate early stopping to prevent overfitting.
- Augment data or fine-tune hyperparameters to optimize results.

---

## Usage
- The saved model and tokenizer are available in `/models/trained_model/` and `/models/tokenizer/`.
- Evaluation metrics and the confusion matrix visualization are in `/reports/`.
- Run `train.py` to retrain the model and `evaluate_saved_model.py` to reproduce the evaluation.

---

## Conclusion
This project successfully demonstrates fine-tuning a transformer-based model for binary sentiment classification on movie reviews, achieving over 90% accuracy. It lays a solid foundation for extending to other NLP classification tasks.
