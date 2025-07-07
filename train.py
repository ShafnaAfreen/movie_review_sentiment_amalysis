# train.py

from data_preprocessing import get_dataset
from train_model import train_model

def main():
    dataset = get_dataset(subset_size=5000)
    trainer = train_model(dataset["train"], dataset["test"])
    
    # Evaluate model
    results = trainer.evaluate()
    print("Evaluation results:", results)

if __name__ == "__main__":
    main()
