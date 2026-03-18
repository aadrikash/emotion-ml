"""
Utility functions for the pipeline.
"""
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import pickle


class FileManager:
    """Handle file I/O operations."""
    
    @staticmethod
    def ensure_dirs(*dirs):
        """Create directories if they don't exist."""
        for d in dirs:
            os.makedirs(d, exist_ok=True)
    
    @staticmethod
    def save_df(df: pd.DataFrame, path: str):
        """Save dataframe to CSV."""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        df.to_csv(path, index=False)
        print(f"✓ Saved to {path}")
    
    @staticmethod
    def load_df(path: str) -> pd.DataFrame:
        """Load dataframe from CSV."""
        return pd.read_csv(path)
    
    @staticmethod
    def save_model(model, path: str):
        """Save model using pickle."""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        print(f"✓ Saved model to {path}")
    
    @staticmethod
    def load_model(path: str):
        """Load model from pickle."""
        with open(path, 'rb') as f:
            return pickle.load(f)


class Logger:
    """Simple logging utility."""
    
    def __init__(self, log_file: str = None):
        self.log_file = log_file
    
    def log(self, message: str, level: str = "INFO"):
        """Log message to console and file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] [{level}] {message}"
        
        print(log_message)
        
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(log_message + '\n')


class MetricsCalculator:
    """Calculate evaluation metrics."""
    
    @staticmethod
    def calculate_accuracy(y_true, y_pred):
        """Calculate accuracy."""
        return np.mean(y_true == y_pred)
    
    @staticmethod
    def calculate_f1(y_true, y_pred, labels=None):
        """Calculate F1 scores."""
        from sklearn.metrics import f1_score
        if labels:
            return f1_score(y_true, y_pred, average='weighted', labels=labels)
        return f1_score(y_true, y_pred, average='weighted')
    
    @staticmethod
    def confusion_matrix(y_true, y_pred):
        """Get confusion matrix."""
        from sklearn.metrics import confusion_matrix
        return confusion_matrix(y_true, y_pred)
    
    @staticmethod
    def classification_report(y_true, y_pred, labels=None):
        """Get detailed classification report."""
        from sklearn.metrics import classification_report
        return classification_report(y_true, y_pred, labels=labels, zero_division=0)