import pandas as pd
import numpy as np
import os
import json
import logging
from datetime import datetime
from pathlib import Path
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, Bidirectional
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_lstm(
    input_path: str, 
    model_path: str,
    max_words: int = 20000,
    max_len: int = 100,
    embedding_dim: int = 128,
    lstm_units: int = 128,
    dense_units: int = 64,
    dropout_rate: float = 0.3,
    batch_size: int = 64,
    epochs: int = 30,
    learning_rate: float = 0.001
):
    """
    Train an LSTM model for sentiment analysis with improved architecture.
    
    Args:
        input_path: Path to the processed CSV file
        model_path: Path to save the trained model
        max_words: Maximum number of words in the vocabulary
        max_len: Maximum length of input sequences
        embedding_dim: Dimension of the embedding layer
        lstm_units: Number of units in the LSTM layer
        dense_units: Number of units in the dense layer
        dropout_rate: Dropout rate
        batch_size: Batch size for training
        epochs: Number of training epochs
        learning_rate: Learning rate for the optimizer
    """
    start_time = datetime.now()
    logger.info("Starting model training...")
    
    try:
        # Create model directory if it doesn't exist
        model_dir = os.path.dirname(model_path)
        os.makedirs(model_dir, exist_ok=True)
        
        # Load and prepare data
        logger.info("Loading and preparing data...")
        df = pd.read_csv(input_path)
        X = df["clean_text"].astype(str)
        y = df["label"]

        # Train/validation/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Further split training data into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )

        # Tokenization
        logger.info("Tokenizing text data...")
        tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
        tokenizer.fit_on_texts(X_train)
        
        # Convert text to sequences and pad
        X_train_seq = tokenizer.texts_to_sequences(X_train)
        X_val_seq = tokenizer.texts_to_sequences(X_val)
        X_test_seq = tokenizer.texts_to_sequences(X_test)
        
        X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding="post", truncating="post")
        X_val_pad = pad_sequences(X_val_seq, maxlen=max_len, padding="post", truncating="post")
        X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding="post", truncating="post")

        # One-hot encode labels
        num_classes = len(np.unique(y))
        y_train_cat = to_categorical(y_train, num_classes=num_classes)
        y_val_cat = to_categorical(y_val, num_classes=num_classes)
        y_test_cat = to_categorical(y_test, num_classes=num_classes)
        
        # Save tokenizer
        tokenizer_path = os.path.join(model_dir, 'tokenizer.json')
        with open(tokenizer_path, 'w', encoding='utf-8') as f:
            json.dump(tokenizer.to_json(), f, ensure_ascii=False)
        logger.info(f"Tokenizer saved to {tokenizer_path}")

        # Build enhanced model
        logger.info("Building model...")
        model = Sequential([
            Embedding(
                input_dim=max_words, 
                output_dim=embedding_dim, 
                input_length=max_len,
                mask_zero=True
            ),
            Bidirectional(LSTM(
                lstm_units, 
                return_sequences=False, 
                dropout=dropout_rate, 
                recurrent_dropout=dropout_rate*0.5
            )),
            Dropout(dropout_rate),
            Dense(dense_units, activation="relu"),
            Dropout(dropout_rate*0.5),
            Dense(num_classes, activation="softmax")
        ])

        # Compile model
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            loss="categorical_crossentropy", 
            optimizer=optimizer, 
            metrics=["accuracy", "AUC", "Precision", "Recall"]
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=model_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        model.summary()

        # Train model
        logger.info("Starting model training...")
        history = model.fit(
            X_train_pad, 
            y_train_cat,
            validation_data=(X_val_pad, y_val_cat),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        # Save the final model
        model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_loss, test_accuracy, test_auc, test_precision, test_recall = model.evaluate(X_test_pad, y_test_cat, verbose=0)
        
        # Generate predictions
        y_pred = model.predict(X_test_pad)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_test_cat, axis=1)
        
        # Generate and save classification report
        report = classification_report(y_true, y_pred_classes, target_names=[f'Class {i}' for i in range(num_classes)])
        logger.info("\nClassification Report:")
        logger.info(report)
        
        # Save classification report
        report_path = os.path.join(model_dir, 'classification_report.txt')
        with open(report_path, 'w') as f:
            f.write(str(report))
        
        # Generate and save confusion matrix
        cm = confusion_matrix(y_true, y_pred_classes)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        cm_path = os.path.join(model_dir, 'confusion_matrix.png')
        plt.savefig(cm_path)
        plt.close()
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        
        # Plot training & validation accuracy values
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        # Plot training & validation loss values
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.tight_layout()
        history_path = os.path.join(model_dir, 'training_history.png')
        plt.savefig(history_path)
        plt.close()
        
        logger.info(f"Training completed in {datetime.now() - start_time}")
        logger.info(f"Test Accuracy: {test_accuracy:.4f}")
        logger.info(f"Test AUC: {test_auc:.4f}")
        logger.info(f"Test Precision: {test_precision:.4f}")
        logger.info(f"Test Recall: {test_recall:.4f}")
        logger.info(f"Model and training artifacts saved to {model_dir}")
        
        return model, history
        
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train an LSTM model for sentiment analysis')
    parser.add_argument('--input', type=str, default='data/processed/cleaned.csv',
                      help='Path to the processed CSV file')
    parser.add_argument('--output', type=str, default='models/lstm_model.keras',
                      help='Path to save the trained model (must end with .keras)')
    parser.add_argument('--max-words', type=int, default=20000,
                      help='Maximum number of words in the vocabulary')
    parser.add_argument('--max-len', type=int, default=100,
                      help='Maximum length of input sequences')
    parser.add_argument('--embedding-dim', type=int, default=128,
                      help='Dimension of the embedding layer')
    parser.add_argument('--lstm-units', type=int, default=128,
                      help='Number of units in the LSTM layer')
    parser.add_argument('--dense-units', type=int, default=64,
                      help='Number of units in the dense layer')
    parser.add_argument('--dropout-rate', type=float, default=0.3,
                      help='Dropout rate')
    parser.add_argument('--batch-size', type=int, default=64,
                      help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30,
                      help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                      help='Learning rate for the optimizer')
    
    args = parser.parse_args()
    
    train_lstm(
        input_path=args.input,
        model_path=args.output,
        max_words=args.max_words,
        max_len=args.max_len,
        embedding_dim=args.embedding_dim,
        lstm_units=args.lstm_units,
        dense_units=args.dense_units,
        dropout_rate=args.dropout_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate
    )
