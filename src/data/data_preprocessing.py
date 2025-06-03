import numpy as np
import pandas as pd

import os
import logging

import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer


# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Set up logging
logger.info("Starting data preprocessing process...") 

# Create handlers for logging to console and file
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('data_preprocessing.log')
file_handler.setLevel(logging.DEBUG)  
formator = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formator)
file_handler.setFormatter(formator)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads train and test data from the given file paths.

    Args:
        train_path (str): Path to the train data CSV file.
        test_path (str): Path to the test data CSV file.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Loaded train and test DataFrames.

    Raises:
        FileNotFoundError: If a file is not found.
        pd.errors.EmptyDataError: If a file is empty.
        Exception: For any other errors during loading.
    """
    try:
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        logger.info("Successfully loaded train and test data.")
        return train_data, test_data
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except pd.errors.EmptyDataError as e:
        logger.error(f"Empty data file: {e}")
        raise
    except Exception as e:
        logger.error(f"An error occurred while loading data: {e}")
        raise

# Usage
train_data, test_data = load_data('./data/raw/train_data.csv', './data/raw/test_data.csv')



# This script is responsible for preprocessing the data, including text normalization, lemmatization, and removing stop words.

# transform the data
try:
    nltk.download('wordnet')
    nltk.download('stopwords')
    logger.info("Successfully downloaded NLTK resources: wordnet and stopwords.")
except Exception as e:
    logger.error(f"An error occurred while downloading NLTK resources: {e}")
    raise


# Function to perform lemmatization
def lemmatization(text):
    try:
        lemmatizer = WordNetLemmatizer()
        text = text.split()
        text = [lemmatizer.lemmatize(y) for y in text]
        return " ".join(text)
    except Exception as e:
        logger.error(f"Error during lemmatization: {e}")
        return text if isinstance(text, str) else " ".join(text)


# Function to remove stop words
def remove_stop_words(text):
    try:
        stop_words = set(stopwords.words("english"))
        Text = [i for i in str(text).split() if i not in stop_words]
        return " ".join(Text)
    except Exception as e:
        logger.error(f"Error during removing stop words: {e}")
        return text if isinstance(text, str) else " ".join(text)


# Function to remove numbers from the text
def removing_numbers(text):
    try:
        text = ''.join([i for i in text if not i.isdigit()])
        return text
    except Exception as e:
        logger.error(f"Error during removing numbers: {e}")
        return text if isinstance(text, str) else ""


# Function to convert text to lower case
def lower_case(text):
    try:
        text = text.split()
        text = [y.lower() for y in text]
        return " ".join(text)
    except Exception as e:
        logger.error(f"Error during converting to lower case: {e}")
        return text if isinstance(text, str) else ""



# Function to remove punctuations from the text
def removing_punctuations(text):
    try:
        # Remove punctuations
        text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
        text = text.replace('؛', "", )
        # remove extra whitespace
        text = re.sub('\s+', ' ', text)
        text = " ".join(text.split())
        return text.strip()
    except Exception as e:
        logger.error(f"Error during removing punctuations: {e}")
        return text if isinstance(text, str) else ""


# Function to remove URLs from the text
def removing_urls(text):
    try:
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)
    except Exception as e:
        logger.error(f"Error during removing URLs: {e}")
        return text if isinstance(text, str) else ""

# Function to remove small sentences (less than 3 words) with exception handling and logging
def remove_small_sentences(df):
    try:
        for i in range(len(df)):
            try:
                if len(df.text.iloc[i].split()) < 3:
                    df.text.iloc[i] = np.nan
            except Exception as e:
                logger.error(f"Error processing row {i}: {e}")
        logger.info("Successfully removed small sentences.")
    except Exception as e:
        logger.error(f"Error in remove_small_sentences: {e}")
        raise


# Normalising text function that applies all preprocessing steps
def normalize_text(df):
    try:
        df.content = df.content.apply(lambda content: lower_case(content))
        df.content = df.content.apply(lambda content: remove_stop_words(content))
        df.content = df.content.apply(lambda content: removing_numbers(content))
        df.content = df.content.apply(lambda content: removing_punctuations(content))
        df.content = df.content.apply(lambda content: removing_urls(content))
        df.content = df.content.apply(lambda content: lemmatization(content))
        logger.info("Text normalization completed successfully.")
        return df
    except Exception as e:
        logger.error(f"Error during text normalization: {e}")
        raise

try:
    # Normalising the train and test data
    train_processed_data = normalize_text(train_data)
    test_processed_data = normalize_text(test_data)
    logger.info("Train and test data normalized successfully.")
except Exception as e:
    logger.error(f"Error during normalization of train or test data: {e}")
    raise


try:
    # store the data inside data/processed
    data_path = os.path.join('data', 'processed')
    os.makedirs(data_path, exist_ok=True)

    train_processed_data.to_csv(os.path.join(data_path, 'train_processed.csv'), index=False)
    test_processed_data.to_csv(os.path.join(data_path, 'test_processed.csv'), index=False)
    logger.info("Processed data successfully saved to data/processed directory.")
except Exception as e:
    logger.error(f"Error while saving processed data: {e}")
    raise

def main():
    try:
        train_data, test_data = load_data('./data/raw/train_data.csv', './data/raw/test_data.csv')
        train_processed_data = normalize_text(train_data)
        test_processed_data = normalize_text(test_data)
        data_path = os.path.join('data', 'processed')
        os.makedirs(data_path, exist_ok=True)
        train_processed_data.to_csv(os.path.join(data_path, 'train_processed.csv'), index=False)
        test_processed_data.to_csv(os.path.join(data_path, 'test_processed.csv'), index=False)
        logger.info("Main function executed successfully.")
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise

if __name__ == "__main__":
    main()