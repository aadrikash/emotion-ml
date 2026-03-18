"""
Data preprocessing and feature engineering for emotion detection.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import re
from typing import Dict
import warnings
warnings.filterwarnings('ignore')


class TextPreprocessor:
    """Process and clean journal text data."""
    
    def __init__(self):
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'been', 'be',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'i', 'me', 'my', 'we', 'you', 'he',
            'she', 'it', 'that', 'this', 'what', 'which', 'who', 'when', 'where',
            'why', 'how'
        }
    
    def clean_text(self, text):
        """Clean and normalize text."""
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def extract_text_features(self, text: str) -> Dict[str, float]:
        """Extract numerical features from text."""
        clean_text = self.clean_text(text)
        
        features = {
            'text_length': len(clean_text),
            'word_count': len(clean_text.split()),
            'avg_word_length': len(clean_text) / max(len(clean_text.split()), 1),
            'has_contradictions': float('but' in clean_text or 'yet' in clean_text),
            'has_uncertainty': float(any(word in clean_text for word in ['idk', 'not sure', 'maybe', 'kinda', 'somehow'])),
        }
        
        return features


class DataPreprocessor:
    """Main preprocessing pipeline."""
    
    def __init__(self):
        self.text_processor = TextPreprocessor()
        self.label_encoders = {}
        self.scaler = StandardScaler()
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values intelligently."""
        df = df.copy()
        
        if 'sleep_hours' in df.columns:
            df['sleep_hours'].fillna(df['sleep_hours'].median() or 7, inplace=True)
        
        categorical_cols = ['previous_day_mood', 'face_emotion_hint']
        for col in categorical_cols:
            if col in df.columns:
                df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'unknown', inplace=True)
        
        if 'journal_text' in df.columns:
            df['journal_text'].fillna("", inplace=True)
        
        return df
    
    def extract_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract text features and add to dataframe."""
        df = df.copy()
        
        if 'journal_text' in df.columns:
            text_features_list = df['journal_text'].apply(self.text_processor.extract_text_features)
            text_features_df = pd.DataFrame(text_features_list.tolist())
            df = pd.concat([df, text_features_df], axis=1)
        
        return df
    
    def encode_categorical(self, df: pd.DataFrame, categorical_cols: list, fit: bool = True) -> pd.DataFrame:
        """Encode categorical variables."""
        df = df.copy()
        
        for col in categorical_cols:
            if col in df.columns:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        return df
    
    def scale_features(self, df: pd.DataFrame, numeric_cols: list, fit: bool = True) -> pd.DataFrame:
        """Scale numeric features."""
        df = df.copy()
        
        if fit:
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        else:
            df[numeric_cols] = self.scaler.transform(df[numeric_cols])
        
        return df
    
    def preprocess(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Complete preprocessing pipeline."""
        df = self.handle_missing_values(df)
        df = self.extract_all_features(df)
        
        categorical_cols = ['ambience_type', 'time_of_day', 'previous_day_mood', 'face_emotion_hint', 'reflection_quality']
        df = self.encode_categorical(df, categorical_cols, fit=fit)
        
        numeric_cols = [col for col in df.columns if df[col].dtype in ['float64', 'int64'] 
                       and col not in ['id', 'emotional_state', 'intensity']]
        df = self.scale_features(df, numeric_cols, fit=fit)
        
        return df