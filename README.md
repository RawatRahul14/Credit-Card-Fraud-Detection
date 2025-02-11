# Credit Card Fraud Detection

A machine learning project aimed at detecting fraudulent credit card transactions using machine learning techniques.

## Project Overview

This project implements a fraud detection system that analyzes credit card transactions to identify potentially fraudulent activities. The system uses machine learning models trained on historical transaction data to make predictions about new transactions.

## Installation

### Prerequisites
- Python 3.11+
- pip package manager
- Git

### Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/RawatRahul14/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset:
```bash
python dataset_download.py
```

**Note:** The dataset folder is excluded from version control using `.gitignore`. Running `dataset_download.py` will automatically set up the Data folder in the correct location as referenced by other project files.

## Project Structure

```
Directory structure:
└── rawatrahul14-credit-card-fraud-detection/
    ├── README.md
    ├── app.py
    ├── model_training.py
    ├── research.ipynb
    ├── Lists/
    │   ├── categories.pkl
    │   ├── col_names.pkl
    │   ├── jobs.pkl
    │   └── states.pkl
    └── Model/
        └── logistic_model.pkl
```