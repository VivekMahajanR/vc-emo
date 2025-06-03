import numpy as np
import pandas as pd

import pickle
import json
import os
import logging

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Set up logging
logger.info("Starting data ingestion process...") 

# Create handlers for logging to console and file
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('data_ingestion.log')
file_handler.setLevel(logging.DEBUG)  
formator = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formator)
file_handler.setFormatter(formator)

logger.addHandler(console_handler)
logger.addHandler(file_handler)



def load_model_and_data(model_path: str, test_data_path: str) -> tuple:
    """
    Loads a pickled model and test data from the given paths.

    Args:
        model_path (str): Path to the pickled model file.
        test_data_path (str): Path to the test data CSV file.

    Returns:
        tuple: (clf, X_test, y_test)
    """
    try:
        with open(model_path, 'rb') as model_file:
            clf = pickle.load(model_file)
        logger.info(f"Model loaded successfully from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        raise

    try:
        test_data = pd.read_csv(test_data_path)
        X_test = test_data.iloc[:, :-1].to_numpy()
        y_test = test_data.iloc[:, -1].to_numpy()
        logger.info(f"Test data loaded successfully from {test_data_path}")
    except Exception as e:
        logger.error(f"Failed to load test data from {test_data_path}: {e}")
        raise

    return clf, X_test, y_test

clf, X_test, y_test = load_model_and_data('./data/models/gb_model.pkl', './data/features/test_bow.csv')


def make_predictions(clf, X_test) -> tuple:
    """
    Makes predictions and calculates probabilities using the provided classifier and test data.

    Args:
        clf: Trained classifier with predict and predict_proba methods.
        X_test: Test features as a numpy array.

    Returns:
        tuple: (y_pred, y_pred_proba)
    """
    try:
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        logger.info("Predictions and probabilities calculated successfully.")
        return y_pred, y_pred_proba
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise

y_pred, y_pred_proba = make_predictions(clf, X_test)


def evaluate_model(y_test: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> dict:
    """
    Calculates evaluation metrics and returns them as a dictionary.

    Args:
        y_test (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        y_pred_proba (np.ndarray): Predicted probabilities for the positive class.

    Returns:
        dict: Dictionary containing accuracy, precision, recall, and auc.
    """
    try:
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }
        logger.info("Evaluation metrics calculated successfully.")
        return metrics_dict
    except Exception as e:
        logger.error(f"Error calculating evaluation metrics: {e}")
        raise

metrics_dict = evaluate_model(y_test, y_pred, y_pred_proba)


def save_metrics_to_json(metrics: dict, file_path: str) -> None:
    """
    Saves the evaluation metrics dictionary to a JSON file.

    Args:
        metrics (dict): Dictionary containing evaluation metrics.
        file_path (str): Path to the JSON file where metrics will be saved.
    """
    try:
        with open(file_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Metrics saved successfully to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save metrics to {file_path}: {e}")
        raise

save_metrics_to_json(metrics_dict, './data/models/evaluation_metrics.json')


def main():
    model_path = './data/models/gb_model.pkl'
    test_data_path = './data/features/test_bow.csv'
    metrics_output_path = './data/models/evaluation_metrics.json'

    clf, X_test, y_test = load_model_and_data(model_path, test_data_path)
    y_pred, y_pred_proba = make_predictions(clf, X_test)
    metrics_dict = evaluate_model(y_test, y_pred, y_pred_proba)
    save_metrics_to_json(metrics_dict, metrics_output_path)

if __name__ == "__main__":
    main()