import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
# Logging configuration
logger = logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Rest of your code...

def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(data_url)
        logger.debug('Data loaded from %s', data_url)
        return df

    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise

    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str, create_raw_subfolder: bool = True) -> None:
    """Save the train and test datasets.
    
    Args:
        train_data: Training DataFrame
        test_data: Test DataFrame
        data_path: Path to save data
        create_raw_subfolder: If True, creates a 'raw' subfolder
    """
    try:
        if create_raw_subfolder:
            raw_data_path = os.path.join(data_path, 'raw')
        else:
            raw_data_path = data_path

        # Create the directory if it does not exist
        os.makedirs(raw_data_path, exist_ok=True)

        # Save the train and test data
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)

        logger.debug('Train and test data saved to %s', raw_data_path)

    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise

    
def load_params(params_path: str = "params.yaml") -> dict:
    """Load parameters from a YAML file."""
    try:
        # If you want to always look in project root, you could do:
        import os
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        full_path = os.path.join(root_dir, params_path)
        
        with open(full_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', full_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', full_path)
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data by handling missing values, duplicates, and empty strings."""
    try:
        # Removing missing values
        df.dropna(inplace=True)

        # Removing duplicates
        df.drop_duplicates(inplace=True)

        # Removing rows with empty strings
        df = df[df['clean_comment'].str.strip() != '']

        logging.debug(
            'Data preprocessing   completed: Missing values, duplicates, and empty strings removed'
        )
        return df

    except KeyError as e:
        logging.error('Missing column in the dataframe: %s', e)
        raise

    except Exception as e:
        logging.error('Unexpected error during preprocessing: %s', e)
        raise

def main():
    try:
        # Load parameters from the params.yaml in the root directory
        params = load_params(
            params_path="params.yaml"
        )
        test_size = params['data_ingestion']['test_size']

        # Load data from the specified URL
        df = load_data(
            data_url="https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv"
        )

        # Preprocess the data
        final_df = preprocess_data(df)

        # Split the data into training and testing sets
        train_data, test_data = train_test_split(
            final_df,
            test_size=test_size,
            random_state=42
        )

        # Save the split datasets and create the raw folder if it doesn't exist
        save_data(
            train_data,
            test_data,
            data_path="artifacts/data",
            create_raw_subfolder=False
            )

    except Exception as e:
        logging.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")

if __name__ == "__main__":
    main()