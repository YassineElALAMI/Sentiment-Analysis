import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import Tuple, Dict, Set, Any
import emoji
import logging
import os
from pathlib import Path
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('preprocessing.log')
    ]
)
logger = logging.getLogger(__name__)

# Download required NLTK resources
try:
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download('omw-1.4', quiet=True)
except Exception as e:
    logger.error(f"Error downloading NLTK resources: {e}")
    raise

# Constants
STOPWORDS: Set[str] = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text: str, remove_stopwords: bool = True) -> str:
    """
    Clean and preprocess text data.
    
    Args:
        text: Input text to clean
        remove_stopwords: Whether to remove stopwords
        
    Returns:
        Cleaned text string
    """
    try:
        if not isinstance(text, str) or not text.strip():
            return ""
            
        # Convert emojis to text description
        text = emoji.demojize(text, delimiters=(" ", " "))
        
        # Basic cleaning
        text = text.lower().strip()
        text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # remove URLs
        text = re.sub(r"@\w+", '', text)  # remove mentions
        text = re.sub(r"#", '', text)  # remove hashtag symbol but keep the word
        text = re.sub(r"[^\w\s']", ' ', text)  # keep words, spaces, and apostrophes
        text = re.sub(r"\s+", ' ', text)  # remove extra whitespace
        
        # Tokenization and lemmatization
        words = text.split()
        if remove_stopwords:
            words = [w for w in words if w not in STOPWORDS]
        words = [lemmatizer.lemmatize(w) for w in words]
        
        return " ".join(words).strip()
        
    except Exception as e:
        logger.error(f"Error cleaning text: {e}")
        return ""

def preprocess(
    input_path: str,
    output_dir: str,
    text_column: str = "Text",
    label_column: str = "Sentiment"
) -> Tuple[str, Dict[str, int]]:
    """
    Load dataset, clean text, map labels, and save processed data.
    
    Args:
        input_path: Path to the input CSV file
        output_dir: Directory to save processed data
        text_column: Name of the column containing text data
        label_column: Name of the column containing labels
        
    Returns:
        Tuple of (output_file_path, label_mapping)
    """
    start_time = datetime.now()
    logger.info(f"Starting preprocessing at {start_time}")
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Validate input file
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Load data
        logger.info(f"Loading data from {input_path}")
        try:
            df = pd.read_csv(input_path)
            logger.info(f"Loaded {len(df)} records")
        except Exception as e:
            logger.error(f"Error loading CSV file: {e}")
            raise
        
        # Validate input columns
        required_columns = {text_column, label_column}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        # Clean text
        logger.info("Cleaning text data...")
        df["clean_text"] = df[text_column].astype(str).apply(clean_text)
        
        # Handle empty texts after cleaning
        empty_texts = df["clean_text"].str.strip().eq("").sum()
        if empty_texts > 0:
            logger.warning(f"Found {empty_texts} empty texts after cleaning")
        
        # Map sentiment labels to integers
        unique_labels = sorted(df[label_column].unique())
        label_map = {label: i for i, label in enumerate(unique_labels)}
        df["label"] = df[label_column].map(label_map)
        
        # Save processed data
        output_path = os.path.join(output_dir, f"cleaned_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        df.to_csv(output_path, index=False)
        
        # Save label mapping
        label_map_path = os.path.join(output_dir, "label_mapping.json")
        with open(label_map_path, 'w', encoding='utf-8') as f:
            json.dump(label_map, f, indent=2, ensure_ascii=False)
            
        # Save processing stats
        stats = {
            "processing_time_seconds": (datetime.now() - start_time).total_seconds(),
            "total_records": len(df),
            "unique_labels": len(unique_labels),
            "empty_texts_after_cleaning": int(empty_texts),
            "output_file": output_path,
            "label_mapping_file": label_map_path
        }
        
        stats_path = os.path.join(output_dir, "preprocessing_stats.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Preprocessing completed in {(datetime.now() - start_time).total_seconds():.2f} seconds")
        logger.info(f"Processed data saved to: {output_path}")
        logger.info(f"Label mapping saved to: {label_map_path}")
        
        return output_path, label_map
        
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess text data for sentiment analysis")
    parser.add_argument("--input", type=str, default="data/raw/tweets.csv",
                        help="Path to input CSV file")
    parser.add_argument("--output-dir", type=str, default="data/processed",
                        help="Directory to save processed data")
    parser.add_argument("--text-col", type=str, default="Text",
                        help="Name of the text column in the input CSV")
    parser.add_argument("--label-col", type=str, default="Sentiment",
                        help="Name of the label column in the input CSV")
    
    args = parser.parse_args()
    
    try:
        output_path, label_map = preprocess(
            input_path=args.input,
            output_dir=args.output_dir,
            text_column=args.text_col,
            label_column=args.label_col
        )
        print(f"\nPreprocessing complete!")
        print(f"Output file: {output_path}")
        print(f"Label mapping: {label_map}")
    except Exception as e:
        logger.critical(f"Preprocessing failed: {str(e)}")
        exit(1)
