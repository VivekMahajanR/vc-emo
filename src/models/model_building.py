import numpy as np
import pandas as pd
import yaml
import os
import pickle
from sklearn.metrics import classification_report


import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score
import logging
from typing import Dict, Any

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




with open('params.yaml', 'r') as file:
    params = yaml.safe_load(file)
    n_estimators = params['model_building']['n_estimators']
    learning_rate = params['model_building']['learning_rate']
    max_depth = params['model_building']['max_depth']
 



def load_training_data(file_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads training data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        tuple[np.ndarray, np.ndarray]: Feature matrix X and target vector y.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        pd.errors.ParserError: If there is an error parsing the CSV file.
        Exception: For any other exceptions.
    """
    try:
        train_data = pd.read_csv(file_path)
        X_train = train_data.iloc[:, 0:-1].to_numpy()
        y_train = train_data.iloc[:, -1].to_numpy()
        return X_train, y_train
    except FileNotFoundError as e:
        logging.error(f"Training data file not found: {file_path}")
        raise
    except pd.errors.ParserError as e:
        logging.error(f"Error parsing CSV file: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error loading training data: {e}")
        raise

# fetch the processed data from data/processed
X_train, y_train = load_training_data('./data/features/train_bow.csv')


def train_gradient_boosting_classifier(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    params: Dict[str, Any]
) -> GradientBoostingClassifier:
    """
    Trains a GradientBoostingClassifier with the given training data and parameters.

    Args:
        X_train (np.ndarray): Feature matrix for training.
        y_train (np.ndarray): Target vector for training.
        params (Dict[str, Any]): Parameters for the classifier.

    Returns:
        GradientBoostingClassifier: The trained model.

    Raises:
        Exception: If model training fails.
    """
    try:
        gb_model = GradientBoostingClassifier(
            n_estimators= n_estimators,
            learning_rate=learning_rate,
            max_depth= max_depth,
            random_state=42
        )
        gb_model.fit(X_train, y_train)
        logging.info("GradientBoostingClassifier trained successfully.")
        return gb_model
    except Exception as e:
        logging.error(f"Error training GradientBoostingClassifier: {e}")
        raise

# Train the model using the function
gb_model = train_gradient_boosting_classifier(X_train, y_train, params)




# os.makedirs(data_path, exist_ok=True)
# with open(os.path.join(data_path, 'gb_model.pkl'), 'wb') as f:
#     pickle.dump(gb_model, f)


def save_model(model: GradientBoostingClassifier, model_dir: str = 'data/models', model_filename: str = 'gb_model.pkl') -> None:
    """
    Saves the trained model to a file using pickle.

    Args:
        model (GradientBoostingClassifier): The trained model to save.
        model_dir (str): Directory where the model will be saved.
        model_filename (str): Name of the file to save the model as.

    Raises:
        Exception: If saving the model fails.
    """
    try:
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, model_filename)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logging.info(f"Model saved successfully at {model_path}")
    except Exception as e:
        logging.error(f"Error saving model: {e}")
        raise

# Save the model
save_model(gb_model)

def main() -> None:
    """
    Main function to execute the model training and saving pipeline.

    Handles exceptions and logs errors if any step fails.
    """
    try:
        X_train: np.ndarray
        y_train: np.ndarray
        X_train, y_train = load_training_data('./data/features/train_bow.csv')
        gb_model: GradientBoostingClassifier = train_gradient_boosting_classifier(X_train, y_train, params)
        save_model(gb_model)
        print("Model training and saving completed successfully.")
    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}", exc_info=True)
        print(f"Pipeline execution failed: {e}")

if __name__ == "__main__":
    main()