# Sentiment Analysis on IMDb Dataset

This project fine-tunes a DistilBERT model to perform sentiment analysis on movie reviews from the IMDb dataset.

## Project Structure

- `/src/` - Source code for training, data preprocessing, and utilities
- `/reports/` - Evaluation metrics, confusion matrix, and model report

## How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   
3. Download Models Folder

The models folder is too large to store in this repo.  
Please download it from Google Drive:

https://drive.google.com/drive/folders/1ZQiB384mytewTWiH_gi3wt6XGuIrHCs8?usp=drive_link

Click "Download" on the page to get the entire folder as a ZIP.
## Usage
To preprocess data, train the model, and evaluate, run the scripts in the /src/ directory.
Modify any configurations or hyperparameters inside the config files or script arguments.

## Results
After training, evaluation reports and confusion matrices will be saved in the /reports/ directory.

You can view performance metrics like accuracy, precision, recall, and F1 score.

Notes
This project uses Hugging Face Transformers and Datasets libraries.

Ensure you have Python 3.7+ and sufficient GPU resources for training.

Large files like pretrained models are excluded from the repo due to GitHub limits.

License
Specify your license here (e.g., MIT License).
