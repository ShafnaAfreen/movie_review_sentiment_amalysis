# Project Submission

## Approach

This project uses a pretrained DistilBERT model fine-tuned on the IMDb movie review dataset for binary sentiment classification (positive/negative). Data preprocessing involved tokenization with padding and truncation to a fixed max length to prepare inputs.

## Model Details

- Base Model: DistilBERT-base-uncased  
- Task: Sequence classification with 2 labels  
- Training: 3 epochs, batch size 16, learning rate 2e-5, weight decay 0.01  
- Framework: Hugging Face Transformers Trainer API  

## Results

- Accuracy: 90.3%  
- Precision: 89.3%  
- Recall: 90.9%  
- F1 Score: 90.15%  

The model performs well in distinguishing positive and negative movie reviews, with balanced precision and recall.

## Key Learnings

- Transformer-based models like DistilBERT are effective for NLP classification tasks even with limited data.  
- The Hugging Face Trainer simplifies fine-tuning and evaluation workflows.  
- Proper tokenization and data splitting are critical for model performance.  
- Monitoring metrics like F1 score helps in selecting the best model.  

## Future Improvements

- Increase dataset size for better generalization.  
- Experiment with other transformer architectures (BERT, RoBERTa).  
- Implement early stopping to prevent overfitting.  
- Fine-tune hyperparameters for optimized performance.  
- Add explainability techniques to interpret model predictions.  

---

Thank you!
