# config.py

from transformers import AutoTokenizer

CHECKPOINT = "distilbert-base-uncased-finetuned-sst-2-english"
MODEL_SAVE_DIR = "./results"
MAX_LENGTH = 512
NUM_LABELS = 2
NUM_EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
SEED = 42

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
