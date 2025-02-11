import os
import sys
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
print("Loading dataset...")
data_path = os.path.join("Data", "fraudTrain.csv")

data = pd.read_csv(data_path)
print("Dataset loaded successfully.")

def preprocess_data(dataset: pd.DataFrame, save_data: bool = True):
    """
    Preprocess the dataset including handling missing values, feature engineering,
    and applying SMOTE for class balancing.
    """
    print("Starting preprocessing...")

    # Remove missing values
    dataset = dataset.dropna()
    print("Missing values dropped.")

    # Removing unnecessary columns
    drop_cols = ["cc_num", "Unnamed: 0", "first", "last", "street", "city", 
                 "zip", "lat", "long", "dob", "trans_num", "unix_time", 
                 "merch_long", "merch_lat", "merchant", "trans_date_trans_time"]
    dataset = dataset.drop(columns = drop_cols)
    print("Unnecessary columns dropped.")

    # Sample non-fraudulent transactions to reduce dataset size
    dataset_not_fraud = dataset[dataset["is_fraud"] == 0].sample(200_000).reset_index(drop = True)
    dataset_fraud = dataset[dataset["is_fraud"] == 1]

    dataset_new = pd.concat([dataset_not_fraud, dataset_fraud], axis = 0)
    print("Dataset balanced through sampling.")

    if save_data:
        os.makedirs("Lists", exist_ok = True)
        category = list(dataset_new["category"].unique())
        state = list(dataset_new["state"].unique())
        jobs = list(dataset_new["job"].unique())

        with open("Lists/categories.pkl", "wb") as file:
            pickle.dump(category, file)
        with open("Lists/states.pkl", "wb") as file:
            pickle.dump(state, file)
        with open("Lists/jobs.pkl", "wb") as file:
            pickle.dump(jobs, file)

    # Applying one hot encoding
    data_encoded = pd.get_dummies(dataset_new, columns = ["category", "gender", "state", "job"])
    print("Applied One Hot Encoding.")

    # Split features and target
    X = data_encoded.drop("is_fraud", axis = 1)
    y = data_encoded["is_fraud"]

    x_cols = list(X.columns)
    with open("Lists/col_names.pkl", "wb") as file:
        pickle.dump(x_cols, file)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    print("Data split into train and test sets.")

    # Apply SMOTE to balance the training data
    smote = SMOTE(sampling_strategy = "auto", random_state = 42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print("SMOTE applied to training data.")

    return X_train_resampled, y_train_resampled, X_test, y_test

from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

def model_train(X_train, y_train, X_test, y_test):
    """Train Logistic Regression model and save it to a file."""
    
    print("Starting model training...")
    
    # Initialize model with appropriate parameters
    lr_model = LogisticRegression(random_state = 42,
                                  max_iter = 1000,
                                  n_jobs = -1)

    # Train the model
    lr_model.fit(X_train, y_train)
    print("Model training completed.")

    # Calculate and print training accuracy
    train_accuracy = lr_model.score(X_train, y_train)
    print(f"Training accuracy: {train_accuracy:.4f}")

    # Calculate and print test accuracy
    test_accuracy = lr_model.score(X_test, y_test)
    print(f"Test accuracy: {test_accuracy:.4f}")

    # Evaluate model on the test set
    y_pred = lr_model.predict(X_test)

    # Calculate F1 score, Precision, and Recall
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f"\nF1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    # ROC-AUC Score
    y_prob = lr_model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"\nROC-AUC Score: {roc_auc:.4f}")

    # Save the model
    print("Saving trained model...")

    os.makedirs("Model", exist_ok = True)

    model_path = os.path.join("Model", "logistic_model.pkl")
    with open(model_path, "wb") as file:
        pickle.dump(lr_model, file)

    print(f"Model saved successfully at {model_path}")

    return lr_model


def main():
    """Main function to preprocess data and train the model."""
    
    print("Starting process...")

    try:
        # Preprocess the dataset
        X_train, y_train, X_test, y_test = preprocess_data(data)

        # Train the model
        model = model_train(X_train, y_train, X_test, y_test)

        print("Process completed successfully.")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()