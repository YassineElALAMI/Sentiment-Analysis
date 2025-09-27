import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("evaluation.log")
    ]
)
logger = logging.getLogger(__name__)

def load_data(
    input_path: str, 
    text_col: str = "clean_text", 
    label_col: str = "label"
) -> Tuple[pd.Series, pd.Series]:
    """
    Load and validate evaluation data.
    
    Args:
        input_path: Path to the processed CSV file
        text_col: Name of the text column
        label_col: Name of the label column
        
    Returns:
        Tuple of (text_series, labels)
    """
    try:
        logger.info(f"Loading evaluation data from {input_path}")
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
        logger.error(f"Error loading evaluation data: {e}")
        raise

def load_label_mapping(input_dir: str) -> Dict[int, str]:
    """
    Load and process label mapping from JSON file.
    
    Args:
        input_dir: Directory containing label_mapping.json
        
    Returns:
        Dictionary mapping label indices to names
    """
    try:
        label_map_path = os.path.join(input_dir, "label_mapping.json")
        logger.info(f"Loading label mapping from {label_map_path}")
        
        with open(label_map_path, 'r') as f:
            label_map = json.load(f)
            
        logger.info(f"Original label map: {label_map}")
        
        # Handle different label map formats
        if all(isinstance(k, str) and k.isdigit() for k in label_map.keys()):
            # Case 1: Keys are numeric strings (e.g., {"0": "negative", "1": "positive"})
            processed_map = {int(k): v for k, v in label_map.items()}
        elif all(isinstance(k, str) and not k.isdigit() for k in label_map.keys()):
            # Case 2: Keys are string labels (e.g., {"negative": 0, "positive": 1})
            # Create reverse mapping: index -> label
            reverse_map = {v: k for k, v in label_map.items()}
            processed_map = reverse_map
        else:
            # Case 3: Mixed or unexpected format, create a new mapping
            logger.warning("Unexpected label map format. Creating new mapping...")
            unique_labels = sorted(set(label_map.values()) if isinstance(list(label_map.values())[0], str) else label_map.keys())
            processed_map = {i: label for i, label in enumerate(unique_labels)}
        
        logger.info(f"Processed label map: {processed_map}")
        return processed_map
        
    except Exception as e:
        logger.error(f"Error loading/processing label map: {e}")
        raise

def evaluate_baseline_model(
    model_path: str,
    vectorizer_path: str,
    X_test: pd.Series,
    y_test: pd.Series,
    output_dir: str,
    label_map: Dict[int, str],
    model_name: str = "baseline"
) -> Dict[str, Any]:
    """
    Evaluate a baseline model (e.g., Logistic Regression).
    
    Args:
        model_path: Path to the trained model
        vectorizer_path: Path to the vectorizer
        X_test: Test features (text)
        y_test: True labels
        output_dir: Directory to save evaluation results
        label_map: Mapping from label indices to names
        model_name: Name of the model for saving results
        
    Returns:
        Dictionary of evaluation metrics
    """
    try:
        logger.info(f"Evaluating baseline model: {model_path}")
        
        # Load model and vectorizer
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        
        # Transform test data
        X_test_vec = vectorizer.transform(X_test)
        
        # Generate predictions
        y_pred = model.predict(X_test_vec)
        y_proba = model.predict_proba(X_test_vec) if hasattr(model, "predict_proba") else None
        
        # Calculate metrics
        metrics = calculate_metrics(
            y_test, y_pred, y_proba,
            label_map, model_name, output_dir
        )
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error evaluating baseline model: {e}")
        raise

def evaluate_lstm_model(
    model_path: str,
    tokenizer_path: str,
    X_test: pd.Series,
    y_test: pd.Series,
    output_dir: str,
    label_map: Dict[int, str],
    model_name: str = "lstm",
    max_sequence_length: int = 50
) -> Dict[str, Any]:
    """
    Evaluate an LSTM model.
    
    Args:
        model_path: Path to the trained LSTM model
        tokenizer_path: Path to the tokenizer
        X_test: Test features (text)
        y_test: True labels
        output_dir: Directory to save evaluation results
        label_map: Mapping from label indices to names
        model_name: Name of the model for saving results
        max_sequence_length: Maximum sequence length for padding
        
    Returns:
        Dictionary of evaluation metrics
    """
    try:
        logger.info(f"Evaluating LSTM model: {model_path}")
        
        # Load model and tokenizer
        model = load_model(model_path)
        tokenizer = joblib.load(tokenizer_path)
        
        # Tokenize and pad test data
        X_test_seq = tokenizer.texts_to_sequences(X_test)
        X_test_pad = pad_sequences(X_test_seq, maxlen=max_sequence_length, padding="post")
        
        # Generate predictions
        y_proba = model.predict(X_test_pad)
        y_pred = y_proba.argmax(axis=1)
        
        # Calculate metrics
        metrics = calculate_metrics(
            y_test, y_pred, y_proba,
            label_map, model_name, output_dir
        )
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error evaluating LSTM model: {e}")
        raise

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    label_map: Dict[int, str],
    model_name: str,
    output_dir: str
) -> Dict[str, Any]:
    """
    Calculate and log evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (for ROC/AUC)
        label_map: Mapping from label indices to names
        model_name: Name of the model
        output_dir: Directory to save results
        
    Returns:
        Dictionary of evaluation metrics
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Get precision, recall, f1 - fix the formatting issue
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        # Also get per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        # Classification report
        target_names = [label_map.get(i, f"Class_{i}") for i in sorted(set(y_true))]
        class_report = classification_report(
            y_true, y_pred,
            target_names=target_names,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=target_names,
            yticklabels=target_names
        )
        plt.title(f"Confusion Matrix - {model_name}")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        cm_path = os.path.join(output_dir, f"{model_name}_confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()
        
        # Initialize variables for ROC plots
        roc_path = None
        pr_path = None
        
        # Calculate ROC and PR curves if probabilities are available
        roc_metrics = {}
        if y_proba is not None and len(np.unique(y_true)) > 1:
            try:
                # ROC Curve
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                n_classes = len(np.unique(y_true))
                
                if n_classes == 2:
                    # Binary classification
                    fpr["micro"], tpr["micro"], _ = roc_curve(y_true, y_proba[:, 1])
                    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
                    
                    # Plot ROC curve
                    plt.figure()
                    plt.plot(fpr["micro"], tpr["micro"],
                             label=f'ROC curve (AUC = {roc_auc["micro"]:.2f})')
                    plt.plot([0, 1], [0, 1], 'k--')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title(f'ROC Curve - {model_name}')
                    plt.legend(loc="lower right")
                    roc_path = os.path.join(output_dir, f"{model_name}_roc_curve.png")
                    plt.savefig(roc_path)
                    plt.close()
                    
                    # ROC AUC score
                    roc_metrics["roc_auc"] = roc_auc["micro"]
                    
                # Multi-class ROC curves (one-vs-rest)
                if n_classes > 2:
                    for i in range(n_classes):
                        fpr[i], tpr[i], _ = roc_curve((y_true == i).astype(int), y_proba[:, i])
                        roc_auc[i] = auc(fpr[i], tpr[i])
                    
                    # Compute micro-average ROC curve and ROC area
                    fpr["micro"], tpr["micro"], _ = roc_curve(
                        np.eye(n_classes)[y_true].ravel(),
                        y_proba.ravel()
                    )
                    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
                    
                    # Plot ROC curves
                    plt.figure(figsize=(10, 8))
                    colors = ['blue', 'red', 'green', 'orange', 'purple']
                    for i, color in zip(range(n_classes), colors[:n_classes]):
                        class_name = label_map.get(i, f"Class_{i}")
                        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                                 label=f'{class_name} (AUC = {roc_auc[i]:.2f})')
                    
                    plt.plot([0, 1], [0, 1], 'k--', lw=2)
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title(f'ROC Curves - {model_name}')
                    plt.legend(loc="lower right")
                    roc_path = os.path.join(output_dir, f"{model_name}_roc_curves.png")
                    plt.savefig(roc_path)
                    plt.close()
                    
                    # ROC AUC score (micro-averaged)
                    roc_metrics["roc_auc_micro"] = roc_auc["micro"]
                    roc_metrics["roc_auc_macro"] = np.mean([roc_auc[i] for i in range(n_classes)])
                
                # Precision-Recall Curve
                precision = dict()
                recall = dict()
                average_precision = dict()
                
                if n_classes == 2:
                    precision["micro"], recall["micro"], _ = precision_recall_curve(
                        y_true, y_proba[:, 1])
                    average_precision["micro"] = average_precision_score(
                        y_true, y_proba[:, 1])
                    
                    # Plot PR curve
                    plt.figure()
                    plt.step(recall["micro"], precision["micro"], where='post')
                    plt.xlabel('Recall')
                    plt.ylabel('Precision')
                    plt.ylim([0.0, 1.05])
                    plt.xlim([0.0, 1.0])
                    plt.title(f'Precision-Recall Curve - {model_name}\nAP={average_precision["micro"]:.2f}')
                    pr_path = os.path.join(output_dir, f"{model_name}_pr_curve.png")
                    plt.savefig(pr_path)
                    plt.close()
                    
                    roc_metrics["average_precision"] = average_precision["micro"]
                
            except Exception as e:
                logger.warning(f"Could not calculate ROC/PR curves: {e}")
        
        # Compile all metrics
        metrics = {
            'accuracy': float(accuracy),
            'precision_weighted': float(precision_weighted),
            'recall_weighted': float(recall_weighted),
            'f1_weighted': float(f1_weighted),
            'precision_per_class': [float(p) for p in precision_per_class],
            'recall_per_class': [float(r) for r in recall_per_class],
            'f1_per_class': [float(f) for f in f1_per_class],
            'support_per_class': [int(s) for s in support],
            'classification_report': class_report,
            'confusion_matrix': cm.tolist(),
            'roc_metrics': roc_metrics if roc_metrics else None,
            'plots': {
                'confusion_matrix': cm_path,
                'roc_curve': roc_path,
                'pr_curve': pr_path
            }
        }
        
        # Save metrics to file
        metrics_path = os.path.join(output_dir, f"{model_name}_evaluation_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Evaluation complete. Metrics saved to {metrics_path}")
        
        # Print summary
        print(f"\n=== {model_name.upper()} Evaluation Results ===")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision (weighted): {precision_weighted:.4f}")
        print(f"Recall (weighted): {recall_weighted:.4f}")
        print(f"F1 Score (weighted): {f1_weighted:.4f}")
        
        # Print per-class metrics
        print("\nPer-class metrics:")
        for i, class_name in enumerate(target_names):
            if i < len(precision_per_class):
                print(f"  {class_name}: P={precision_per_class[i]:.4f}, R={recall_per_class[i]:.4f}, F1={f1_per_class[i]:.4f}")
        
        if roc_metrics:
            print("\nROC/AUC metrics:")
            for name, value in roc_metrics.items():
                print(f"  {name.replace('_', ' ').title()}: {value:.4f}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        raise

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate sentiment analysis models")
    
    # Required arguments
    parser.add_argument(
        "--input", 
        type=str, 
        default="data/processed/cleaned.csv",
        help="Path to processed evaluation data CSV (default: data/processed/cleaned.csv)"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="evaluation_results",
        help="Directory to save evaluation results"
    )
    
    # Model arguments
    model_group = parser.add_argument_group('Model Arguments')
    model_group.add_argument(
        "--baseline-model", 
        type=str, 
        default="models/baseline/sentiment_baseline_model_20250927_193227.pkl",
        help="Path to baseline model (default: models/baseline/sentiment_baseline_model_20250927_193227.pkl)"
    )
    model_group.add_argument(
        "--vectorizer", 
        type=str, 
        default="models/baseline/sentiment_baseline_vectorizer_20250927_193227.pkl",
        help="Path to TF-IDF vectorizer (default: models/baseline/sentiment_baseline_vectorizer_20250927_193227.pkl)"
    )
    model_group.add_argument(
        "--lstm-model", 
        type=str, 
        default="models/lstm_model.keras",
        help="Path to LSTM model (default: models/lstm_model.keras)"
    )
    model_group.add_argument(
        "--tokenizer", 
        type=str, 
        default="models/tokenizer.json",
        help="Path to tokenizer (default: models/tokenizer.json)"
    )
    
    # Data arguments
    data_group = parser.add_argument_group('Data Arguments')
    data_group.add_argument(
        "--text-col", 
        type=str, 
        default="clean_text",
        help="Name of the text column in the input CSV"
    )
    data_group.add_argument(
        "--label-col", 
        type=str, 
        default="label",
        help="Name of the label column in the input CSV"
    )
    data_group.add_argument(
        "--label-map", 
        type=str, 
        help="Path to label mapping JSON file (default: same directory as input file)"
    )
    
    # Evaluation options
    eval_group = parser.add_argument_group('Evaluation Options')
    eval_group.add_argument(
        "--max-seq-length", 
        type=int, 
        default=50,
        help="Maximum sequence length for LSTM (default: 50)"
    )
    
    return parser.parse_args()

def main():
    """Main evaluation function."""
    try:
        args = parse_arguments()
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Load data
        X, y = load_data(args.input, args.text_col, args.label_col)
        
        # Load label mapping
        if args.label_map:
            label_map_path = os.path.dirname(args.label_map)
            label_map = load_label_mapping(label_map_path)
        else:
            # Default to same directory as input file
            input_dir = os.path.dirname(args.input)
            label_map = load_label_mapping(input_dir)
        
        # Evaluate models
        results = {}
        
        # Evaluate baseline model if provided
        if args.baseline_model:
            if not args.vectorizer:
                raise ValueError("Vectorizer path is required for baseline model evaluation")
                
            results['baseline'] = evaluate_baseline_model(
                model_path=args.baseline_model,
                vectorizer_path=args.vectorizer,
                X_test=X,
                y_test=y,
                output_dir=args.output_dir,
                label_map=label_map,
                model_name="baseline"
            )
        
        # Evaluate LSTM model if provided
        if args.lstm_model:
            if not args.tokenizer:
                raise ValueError("Tokenizer path is required for LSTM model evaluation")
                
            results['lstm'] = evaluate_lstm_model(
                model_path=args.lstm_model,
                tokenizer_path=args.tokenizer,
                X_test=X,
                y_test=y,
                output_dir=args.output_dir,
                label_map=label_map,
                model_name="lstm",
                max_sequence_length=args.max_seq_length
            )
        
        # Save combined results
        if results:
            combined_path = os.path.join(args.output_dir, "combined_evaluation_results.json")
            with open(combined_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Combined evaluation results saved to {combined_path}")
        
        return 0
        
    except Exception as e:
        logger.critical(f"Evaluation failed: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())