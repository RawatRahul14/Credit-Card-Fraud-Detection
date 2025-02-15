import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
import pickle

def main():
    print("Loading data...")
    data = pd.read_csv("Data/creditcard.csv")

    print("Separating fraud and non-fraud data...")
    fraud_data = data[data["Class"] == 1]
    not_fraud_data = data[data["Class"] == 0].sample(n = 10_000, random_state = 42)

    print("Combining and shuffling datasets...")
    data_new = pd.concat([fraud_data, not_fraud_data], axis = 0).sample(frac = 1, random_state = 42).reset_index(drop = True)

    print("Dropping 'Time' column...")
    data_new.drop("Time", axis = 1, inplace = True)

    print("Splitting into features and target...")
    X = data_new.drop("Class", axis = 1)
    y = data_new["Class"]

    print("Splitting into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

    print("Scaling data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Saving scaler model...")
    with open("Models/scaler.pkl", "wb") as file:
        pickle.dump(scaler, file)

    print("Applying SMOTE to balance training data...")
    smote = SMOTE(random_state = 42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    print("Training logistic regression model...")
    model = LogisticRegression(max_iter = 5000)
    model.fit(X_train_resampled, y_train_resampled)

    print("Making predictions...")
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    print("Calculating performance metrics...")
    print(f"Test Data Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

    print("Saving trained model...")
    with open("Models/fraud_model.pkl", "wb") as file:
        pickle.dump(model, file)

if __name__ == "__main__":
    main()