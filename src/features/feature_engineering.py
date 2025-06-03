import numpy as np
import pandas as pd
import yaml

import os
import logging

from sklearn.feature_extraction.text import CountVectorizer
import joblib

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Set up logging
logger.info("Starting feature Engineering process...") 

# Create handlers for logging to console and file
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('feature_engineering.log')
file_handler.setLevel(logging.DEBUG)  
formator = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formator)
file_handler.setFormatter(formator)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str = 'params.yaml') -> int:
    """
    Loads parameters from a YAML file and returns max_features.
    """
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
            max_features: int = params['feature_engineering']['max_features']
            return max_features
    except FileNotFoundError:
        logger.error(f"{params_path} file not found.")
        raise
    except KeyError as e:
        logger.error(f"Missing key in {params_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading parameters: {e}")
        raise

max_features = load_params('params.yaml')


def load_processed_data(
    train_path: str = './data/processed/train_processed.csv',
    test_path: str = './data/processed/test_processed.csv'
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads processed train and test data from CSV files with exception handling and logging.

    Args:
        train_path (str): Path to the processed training data CSV.
        test_path (str): Path to the processed test data CSV.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Loaded train and test DataFrames.
    """
    try:
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        logger.info("Processed train and test data loaded successfully.")
        return train_data, test_data
    except FileNotFoundError as e:
        logger.error(f"Processed data file not found: {e}")
        raise
    except pd.errors.EmptyDataError as e:
        logger.error(f"Processed data file is empty: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading processed data: {e}")
        raise

# Usage
train_data, test_data = load_processed_data()


# removing missing values
def remove_missing_values(
    train_data: pd.DataFrame, 
    test_data: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        train_data.fillna('', inplace=True)
        test_data.fillna('', inplace=True)
        logger.info("Missing values removed from train and test data.")
    except Exception as e:
        logger.error(f"Error removing missing values: {e}")
        raise
    return train_data, test_data

train_data, test_data = remove_missing_values(train_data, test_data)


def extract_features_and_labels(
    train_data: pd.DataFrame, 
    test_data: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts features and labels from train and test dataframes.
    Returns:
        X_train, y_train, X_test, y_test
    """
    try:
        X_train = train_data['content'].values
        y_train = train_data['sentiment'].values

        X_test = test_data['content'].values
        y_test = test_data['sentiment'].values

        logger.info("Successfully extracted features and labels from train and test data.")
        return X_train, y_train, X_test, y_test
    except KeyError as e:
        logger.error(f"Missing expected column in data: {e}")
        raise
    except Exception as e:
        logger.error(f"Error extracting features and labels: {e}")
        raise

X_train, y_train, X_test, y_test = extract_features_and_labels(train_data, test_data)

def apply_bag_of_words(
    X_train: np.ndarray, 
    X_test: np.ndarray, 
    max_features: int
) -> tuple[np.ndarray, np.ndarray, CountVectorizer]:
    """
    Applies Bag of Words (CountVectorizer) to the training and test data.
    Returns:
        X_train_bow, X_test_bow, vectorizer
    """
    try:
        vectorizer = CountVectorizer(max_features=max_features)
        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)
        logger.info("Bag of Words feature extraction successful.")
        return X_train_bow, X_test_bow, vectorizer
    except Exception as e:
        logger.error(f"Error during Bag of Words feature extraction: {e}")
        raise

X_train_bow, X_test_bow, vectorizer = apply_bag_of_words(X_train, X_test, max_features)


def convert_bow_to_dataframe(
    X_train_bow: np.ndarray, 
    y_train: np.ndarray, 
    X_test_bow: np.ndarray, 
    y_test: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Converts Bag of Words arrays and labels to pandas DataFrames.
    Returns:
        train_df, test_df
    """
    try:
        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df['label'] = y_train

        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df['label'] = y_test

        logger.info("Successfully converted BOW arrays to DataFrames.")
        return train_df, test_df
    except Exception as e:
        logger.error(f"Error converting BOW arrays to DataFrames: {e}")
        raise

train_df, test_df = convert_bow_to_dataframe(X_train_bow, y_train, X_test_bow, y_test)

def save_features_and_vectorizer(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    vectorizer: CountVectorizer,
    data_path: str = os.path.join('data', 'features'),
    vectorizer_filename: str = 'vectorizer.pkl'
) -> None:
    """
    Saves the train and test feature DataFrames and the vectorizer to disk.
    """

    try:
        os.makedirs(data_path, exist_ok=True)
        train_df.to_csv(os.path.join(data_path, 'train_bow.csv'), index=False)
        test_df.to_csv(os.path.join(data_path, 'test_bow.csv'), index=False)
        joblib.dump(vectorizer, os.path.join(data_path, vectorizer_filename))
        logger.info("Features and vectorizer saved successfully.")
    except Exception as e:
        logger.error(f"Error saving features or vectorizer: {e}")
        raise

save_features_and_vectorizer(train_df, test_df, vectorizer)


def save_vectorizer(
    vectorizer: CountVectorizer,
    data_path: str = os.path.join('data', 'features'),
    vectorizer_filename: str = 'vectorizer.pkl'
) -> None:
    """
    Saves the CountVectorizer to a file with exception handling and logging.
    """
    try:
        os.makedirs(data_path, exist_ok=True)
        vectorizer_path = os.path.join(data_path, vectorizer_filename)
        joblib.dump(vectorizer, vectorizer_path)
        logger.info(f"Vectorizer saved successfully at {vectorizer_path}.")
    except Exception as e:
        logger.error(f"Error saving vectorizer: {e}")
        raise

# Call the function to save the vectorizer
save_vectorizer(vectorizer)


def save_bow_dataframes(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    data_path: str = os.path.join('data', 'features')
) -> None:
    """
    Saves the train and test Bag of Words DataFrames to CSV files with exception handling and logging.
    """
    try:
        os.makedirs(data_path, exist_ok=True)
        train_csv_path = os.path.join(data_path, 'train_bow.csv')
        test_csv_path = os.path.join(data_path, 'test_bow.csv')
        train_df.to_csv(train_csv_path, index=False)
        test_df.to_csv(test_csv_path, index=False)
        logger.info(f"Train and test BOW DataFrames saved successfully at {data_path}.")
    except Exception as e:
        logger.error(f"Error saving BOW DataFrames: {e}")
        raise

# Example usage:
save_bow_dataframes(train_df, test_df)


def main() -> None:
    try:
        max_features = load_params('params.yaml')
        train_data, test_data = load_processed_data()
        train_data, test_data = remove_missing_values(train_data, test_data)
        X_train, y_train, X_test, y_test = extract_features_and_labels(train_data, test_data)
        X_train_bow, X_test_bow, vectorizer = apply_bag_of_words(X_train, X_test, max_features)
        train_df, test_df = convert_bow_to_dataframe(X_train_bow, y_train, X_test_bow, y_test)
        save_features_and_vectorizer(train_df, test_df, vectorizer)
        save_vectorizer(vectorizer)
        save_bow_dataframes(train_df, test_df)
        logger.info("Feature engineering pipeline completed successfully.")
    except Exception as e:
        logger.error(f"Feature engineering pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()