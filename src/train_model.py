# train_model.py

from transformers import TrainingArguments, Trainer
from config import MODEL_SAVE_DIR, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY
from model_utils import get_model, compute_metrics

def train_model(train_dataset, eval_dataset):
    training_args = TrainingArguments(
        output_dir=MODEL_SAVE_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        load_best_model_at_end=True,
        metric_for_best_model="f1"
    )

    model = get_model()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    return trainer
