import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

import os
import logging


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


# This script is responsible for ingesting data, preprocessing it, and saving it into train and test sets.

# Load configuration from YAML file
def load_params(params_path: str) -> float:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
            test_size = params['data_ingestion']['test_size']
            logger.info(f"Loaded test_size={test_size} from {params_path}")
            return test_size
    except FileNotFoundError:
        logger.error(f"Parameter file {params_path} not found.")
        raise
    except KeyError as e:
        logger.error(f"Missing key in parameter file: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading parameters: {e}")
        raise


# Read the dataset from the provided URL
def read_data(url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url)
        logger.info(f"Successfully read data from {url}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found at {url}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Parsing error while reading {url}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error reading data from {url}: {e}")
        raise


# Preprocess the dataset
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        # Remove unnecessary columns
        df.drop(columns=['tweet_id'], inplace=True)
        logger.info("Dropped 'tweet_id' column.")

        # Filter for relevant sentiment labels
        final_df = df[df['sentiment'].isin(['happiness', 'sadness'])]
        logger.info("Filtered dataset for 'happiness' and 'sadness' sentiments.")

        # Convert sentiment labels to binary values
        final_df.loc[:, 'sentiment'] = final_df['sentiment'].replace({'happiness': 1, 'sadness': 0})
        logger.info("Converted sentiment labels to binary values.")

        return final_df
    except KeyError as e:
        logger.error(f"Missing column during preprocessing: {e}")
        raise
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise


# Split the data into train and test sets
def split_data(final_df: pd.DataFrame, test_size: float, random_state=42):
    try:
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=random_state)
        logger.info(f"Split data into train (n={len(train_data)}) and test (n={len(test_data)}) sets with test_size={test_size}.")
        return train_data, test_data
    except ValueError as e:
        logger.error(f"ValueError during train/test split: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during train/test split: {e}")
        raise


# Save the train and test data to CSV files
def save_data(data_path='data/raw', train_data=None, test_data=None) -> None:
    try:
        os.makedirs(data_path, exist_ok=True)
        train_path = os.path.join(data_path, 'train_data.csv')
        test_path = os.path.join(data_path, 'test_data.csv')
        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)
        logger.info(f"Train data saved to {train_path}")
        logger.info(f"Test data saved to {test_path}")
        return train_data, test_data
    except Exception as e:
        logger.error(f"Error saving data to CSV files: {e}")
        raise


# Main function to orchestrate the data ingestion process with exception handling and logging
def main():
    try:
        # Load parameters
        test_size = load_params('params.yaml')

        # Read the dataset
        df = read_data('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')

        # Preprocess the dataset
        final_df = preprocess_data(df)

        # Split the data into train and test sets
        train_data, test_data = split_data(final_df, test_size)

        # Save the train and test data
        save_data(train_data=train_data, test_data=test_data)

        logger.info("Data ingestion process completed successfully.")
    except Exception as e:
        logger.error(f"Data ingestion process failed: {e}", exc_info=True)


if __name__ == "__main__":

    main()