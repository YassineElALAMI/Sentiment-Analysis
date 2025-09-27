import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Any, Optional

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training.log")
    ]
)
logger = logging.getLogger(__name__)

def load_data(input_path: str, text_col: str = "clean_text", label_col: str = "label") -> Tuple[pd.Series, pd.Series]:
    """
    Load and validate training data.
    
    Args:
        input_path: Path to the processed CSV file
        text_col: Name of the text column
        label_col: Name of the label column
        
    Returns:
        Tuple of (text_series, labels)
    """
    try:
        logger.info(f"Loading data from {input_path}")
        df = pd.read_csv(input_path)
        
        # Validate columns
        required_columns = {text_col, label_col}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        # Check for empty texts
        if df[text_col].isna().any() or (df[text_col].str.strip() == "").any():
            logger.warning("Found empty or NaN texts in the dataset")
            
        return df[text_col], df[label_col]
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def train_model(
    X_train, 
    y_train,
    model_type: str = "logistic_regression",
    random_state: int = 42,
    cv_folds: int = 5,
    n_jobs: int = -1
) -> Tuple[Any, Dict[str, float]]:
    """
    Train a baseline model with cross-validation.
    
    Args:
        X_train: Training features
        y_train: Training labels
        model_type: Type of model to train (logistic_regression, svm, random_forest)
        random_state: Random seed for reproducibility
        cv_folds: Number of cross-validation folds
        n_jobs: Number of CPU cores to use (-1 for all)
        
    Returns:
        Tuple of (trained_model, cv_metrics)
    """
    try:
        logger.info(f"Training {model_type} model with {cv_folds}-fold CV")
        
        # Model selection
        if model_type == "logistic_regression":
            model = LogisticRegression(
                max_iter=1000,
                random_state=random_state,
                n_jobs=n_jobs,
                class_weight='balanced'
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        cv_scores = cross_val_score(
            model, X_train, y_train, 
            cv=cv, scoring='f1_weighted',
            n_jobs=n_jobs
        )
        
        # Train final model on full training set
        model.fit(X_train, y_train)
        
        cv_metrics = {
            'cv_mean_f1': np.mean(cv_scores),
            'cv_std_f1': np.std(cv_scores),
            'cv_scores': cv_scores.tolist()
        }
        
        logger.info(f"CV F1 Score: {cv_metrics['cv_mean_f1']:.4f} ± {cv_metrics['cv_std_f1']:.4f}")
        
        return model, cv_metrics
        
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise

def evaluate_model(
    model, 
    vectorizer, 
    X_test, 
    y_test,
    output_dir: str,
    label_map: Dict[int, str],
    model_name: str = "baseline"
) -> Dict[str, Any]:
    """
    Evaluate model performance and generate reports.
    
    Args:
        model: Trained model
        vectorizer: Fitted vectorizer
        X_test: Test features
        y_test: True labels
        output_dir: Directory to save evaluation results
        label_map: Mapping from label indices to names
        model_name: Name of the model for saving files
        
    Returns:
        Dictionary of evaluation metrics
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate predictions
        X_test_vec = vectorizer.transform(X_test)
        y_pred = model.predict(X_test_vec)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted', zero_division=0
        )
        
        # Classification report
        report = classification_report(
            y_test, y_pred, 
            target_names=label_map.values(),
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt="d",
            cmap="Blues",
            xticklabels=label_map.values(),
            yticklabels=label_map.values()
        )
        plt.title("Confusion Matrix")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{model_name}_confusion_matrix.png"))
        plt.close()
        
        # Save metrics
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'classification_report': report,
            'confusion_matrix': cm.tolist()
        }
        
        # Save metrics to file
        metrics_path = os.path.join(output_dir, f"{model_name}_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
            
        logger.info(f"Model evaluation complete. Metrics saved to {metrics_path}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        raise

def save_artifacts(
    model, 
    vectorizer, 
    output_dir: str,
    model_name: str = "baseline"
) -> Dict[str, str]:
    """
    Save model artifacts to disk.
    
    Args:
        model: Trained model
        vectorizer: Fitted vectorizer
        output_dir: Directory to save artifacts
        model_name: Base name for saved files
        
    Returns:
        Dictionary of saved file paths
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate unique filenames with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(output_dir, f"{model_name}_model_{timestamp}.pkl")
        vectorizer_path = os.path.join(output_dir, f"{model_name}_vectorizer_{timestamp}.pkl")
        
        # Save model and vectorizer
        joblib.dump(model, model_path)
        joblib.dump(vectorizer, vectorizer_path)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Vectorizer saved to {vectorizer_path}")
        
        return {
            'model_path': model_path,
            'vectorizer_path': vectorizer_path
        }
        
    except Exception as e:
        logger.error(f"Error saving model artifacts: {e}")
        raise

def train_baseline(
    input_path: str,
    output_dir: str,
    model_name: str = "baseline",
    test_size: float = 0.2,
    random_state: int = 42,
    max_features: int = 10000,
    cv_folds: int = 5,
    n_jobs: int = -1
) -> Dict[str, Any]:
    """
    End-to-end training pipeline for baseline model.
    
    Args:
        input_path: Path to processed CSV file
        output_dir: Directory to save model artifacts and metrics
        model_name: Name for the trained model
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        max_features: Maximum number of TF-IDF features
        cv_folds: Number of cross-validation folds
        n_jobs: Number of CPU cores to use (-1 for all)
        
    Returns:
        Dictionary containing training results and paths to artifacts
    """
    start_time = datetime.now()
    logger.info(f"Starting training at {start_time}")
    
    try:
        # Load and prepare data
        X, y = load_data(input_path)
        
        # Load and process label mapping
        label_map_path = os.path.join(os.path.dirname(input_path), "label_mapping.json")
        try:
            with open(label_map_path, 'r') as f:
                label_map = json.load(f)
            
            logger.info(f"Original label map: {label_map}")
            
            # Create a new mapping that ensures integer keys
            if all(isinstance(k, str) and k.isdigit() for k in label_map.keys()):
                # Case 1: Keys are numeric strings (e.g., {"0": "negative", "1": "positive"})
                label_map = {int(k): v for k, v in label_map.items()}
            elif all(isinstance(k, str) and not k.isdigit() for k in label_map.keys()):
                # Case 2: Keys are string labels (e.g., {"negative": 0, "positive": 1})
                # Create a new mapping with sequential integers
                unique_labels = sorted(label_map.values())
                label_map = {i: label for i, label in enumerate(unique_labels)}
            else:
                # Case 3: Mixed or unexpected format, create a new mapping
                logger.warning("Unexpected label map format. Creating new mapping...")
                unique_labels = sorted(set(label_map.values()))
                label_map = {i: label for i, label in enumerate(unique_labels)}
            
            logger.info(f"Processed label map: {label_map}")
            
            # Ensure labels in y are integers
            y = y.astype(int)
            
        except Exception as e:
            logger.error(f"Error loading/processing label map: {e}")
            raise
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=y
        )
        
        # Vectorize text
        logger.info("Fitting TF-IDF vectorizer")
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),  # Include bigrams
            min_df=5,            # Ignore terms that appear in fewer than 5 documents
            max_df=0.8,          # Ignore terms that appear in more than 80% of documents
            stop_words='english'
        )
        X_train_vec = vectorizer.fit_transform(X_train)
        
        # Train model
        model, cv_metrics = train_model(
            X_train_vec, y_train,
            model_type="logistic_regression",
            random_state=random_state,
            cv_folds=cv_folds,
            n_jobs=n_jobs
        )
        
        # Evaluate on test set
        test_metrics = evaluate_model(
            model, vectorizer, X_test, y_test,
            output_dir=output_dir,
            label_map=label_map,
            model_name=model_name
        )
        
        # Save model artifacts
        artifacts = save_artifacts(
            model, vectorizer,
            output_dir=output_dir,
            model_name=model_name
        )
        
        # Prepare results
        training_time = (datetime.now() - start_time).total_seconds()
        results = {
            'model_name': model_name,
            'training_time_seconds': training_time,
            'cv_metrics': cv_metrics,
            'test_metrics': test_metrics,
            'artifacts': artifacts,
            'parameters': {
                'test_size': test_size,
                'random_state': random_state,
                'max_features': max_features,
                'cv_folds': cv_folds
            }
        }
        
        # Save full results
        results_path = os.path.join(output_dir, f"{model_name}_training_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Results saved to {results_path}")
        
        return results
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a baseline sentiment analysis model")
    parser.add_argument(
        "--input", 
        type=str, 
        default="data/processed/cleaned.csv",
        help="Path to processed training data"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="models/baseline",
        help="Directory to save model artifacts"
    )
    parser.add_argument(
        "--model-name", 
        type=str, 
        default="sentiment_baseline",
        help="Name for the trained model"
    )
    parser.add_argument(
        "--test-size", 
        type=float, 
        default=0.2,
        help="Proportion of data to use for testing"
    )
    parser.add_argument(
        "--max-features", 
        type=int, 
        default=10000,
        help="Maximum number of TF-IDF features"
    )
    parser.add_argument(
        "--cv-folds", 
        type=int, 
        default=5,
        help="Number of cross-validation folds"
    )
    parser.add_argument(
        "--n-jobs", 
        type=int, 
        default=-1,
        help="Number of CPU cores to use (-1 for all)"
    )
    parser.add_argument(
        "--random-state", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    return parser.parse_args()

if __name__ == "__main__":
    try:
        args = parse_arguments()
        
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Train the model
        results = train_baseline(
            input_path=args.input,
            output_dir=args.output_dir,
            model_name=args.model_name,
            test_size=args.test_size,
            random_state=args.random_state,
            max_features=args.max_features,
            cv_folds=args.cv_folds,
            n_jobs=args.n_jobs
        )
        
        # Print summary
        print("\nTraining completed successfully!")
        print(f"Model saved to: {results['artifacts']['model_path']}")
        print(f"Test Accuracy: {results['test_metrics']['accuracy']:.4f}")
        print(f"Test F1 Score: {results['test_metrics']['f1']:.4f}")
        
    except Exception as e:
        logger.critical(f"Script failed: {e}", exc_info=True)
        sys.exit(1)
