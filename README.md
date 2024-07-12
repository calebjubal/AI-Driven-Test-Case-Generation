## Overview

This project involves developing an AI model to generate test cases from software requirements using natural language processing (NLP) techniques. The project includes data preprocessing, model development and training, evaluation, and deployment using Flask. The final model is deployed as a web service, accessible via a Flask API.

## Project Structure

- `data_preprocessing.py`: Script for preprocessing the requirements data.
- `train_model.py`: Script for training the BERT model on the preprocessed data.
- `integrate_model.py`: Script for testing the integration of the trained model.
- `app.py`: Flask application to deploy the model as a web service.
- `requirements_dataset.csv`: Sample dataset containing software requirements.
- `processed_requirements.csv`: Preprocessed dataset.
- `results/`: Directory to store the trained model.
- `README.md`: Project documentation.

## Setup Instructions

### 1. Setting Up the Environment

Install the required libraries using pip:

```sh
pip install numpy pandas scikit-learn transformers torch flask
2. Data Collection and Preprocessing
data_preprocessing.py:

python
Copy code
import pandas as pd

# Load the dataset
df = pd.read_csv('requirements_dataset.csv')

# Extract relevant columns
requirements = df[df['Type'].isin(['F', 'US'])]['Requirement']

# Preprocess the data
def preprocess_text(text):
    text = text.lower()  # Convert text to lowercase
    text = ''.join(e for e in text if e.isalnum() or e.isspace())  # Remove special characters
    return text

requirements = requirements.apply(preprocess_text)

# Save the processed data
requirements.to_csv('processed_requirements.csv', index=False)
