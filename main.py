"""
Main pipeline: End-to-end emotion detection and wellness guidance system.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from src.preprocessing import DataPreprocessor
from src.models import EmotionalStateModel, IntensityModel
from src.decision_engine import DecisionEngine
from src.uncertainty import UncertaintyQuantifier
from src.utils import FileManager, Logger, MetricsCalculator
import warnings
warnings.filterwarnings('ignore')


class EmotionDetectionPipeline:
    """Complete ML pipeline for emotion detection."""
    
    def __init__(self, log_file: str = 'results/pipeline.log'):
        self.logger = Logger(log_file)
        self.preprocessor = DataPreprocessor()
        self.emotion_model = EmotionalStateModel()
        self.intensity_model = IntensityModel()
        self.decision_engine = DecisionEngine()
        self.uncertainty_quantifier = UncertaintyQuantifier()
        self.metrics = MetricsCalculator()
        
        # Ensure output directories exist
        FileManager.ensure_dirs('data', 'models', 'results', 'notebooks')
    
    def print_header(self, title: str):
        """Print formatted header."""
        print("\n" + "="*80)
        print(f"🚀 {title}")
        print("="*80 + "\n")
    
    def load_data(self, train_path: str, test_path: str):
        """Load training and test data."""
        self.print_header("LOADING DATA")
        
        try:
            self.train_data = FileManager.load_df(train_path)
            self.test_data = FileManager.load_df(test_path)
            
            print(f"✓ Training data: {self.train_data.shape}")
            print(f"✓ Test data: {self.test_data.shape}")
            
            # Get unique emotional states
            if 'emotional_state' in self.train_data.columns:
                unique_states = self.train_data['emotional_state'].unique()
                print(f"✓ Emotional states: {len(unique_states)} unique states")
                print(f"  States: {', '.join(unique_states)}")
            
            # Get intensity range
            if 'intensity' in self.train_data.columns:
                min_intensity = self.train_data['intensity'].min()
                max_intensity = self.train_data['intensity'].max()
                print(f"✓ Intensity range: {min_intensity} - {max_intensity}")
            
        except Exception as e:
            self.logger.log(f"Error loading data: {e}", "ERROR")
            raise
    
    def preprocess_data(self):
        """Preprocess training and test data."""
        self.print_header("PREPROCESSING DATA")
        
        try:
            # Preprocess training data (fit)
            self.train_processed = self.preprocessor.preprocess(self.train_data, fit=True)
            
            # Preprocess test data (transform only)
            self.test_processed = self.preprocessor.preprocess(self.test_data, fit=False)
            
            # Count features
            feature_cols = [col for col in self.train_processed.columns 
                           if col not in ['id', 'emotional_state', 'intensity', 'journal_text']]
            print(f"✓ Processed features: {len(feature_cols)}")
            
        except Exception as e:
            self.logger.log(f"Error preprocessing data: {e}", "ERROR")
            raise
    
    def train_models(self):
        """Train emotion and intensity models."""
        self.print_header("TRAINING EMOTIONAL STATE MODEL")
        
        try:
            # Prepare features
            feature_cols = [col for col in self.train_processed.columns 
                           if col not in ['id', 'emotional_state', 'intensity', 'journal_text']]
            X_train = self.train_processed[feature_cols]
            y_train_emotion = self.train_processed['emotional_state']
            
            # Train emotion model
            self.emotion_model.fit(X_train, y_train_emotion)
            print(f"✓ Emotion model trained")
            
        except Exception as e:
            self.logger.log(f"Error training emotion model: {e}", "ERROR")
            raise
        
        self.print_header("TRAINING INTENSITY MODEL")
        
        try:
            # Train intensity model
            y_train_intensity = self.train_processed['intensity']
            self.intensity_model.fit(X_train, y_train_intensity)
            print(f"✓ Intensity model trained")
            
        except Exception as e:
            self.logger.log(f"Error training intensity model: {e}", "ERROR")
            raise
    
    def make_predictions(self):
        """Make predictions on test data."""
        self.print_header("MAKING PREDICTIONS ON TEST DATA")
        
        try:
            # Prepare features
            feature_cols = [col for col in self.test_processed.columns 
                           if col not in ['id', 'emotional_state', 'intensity', 'journal_text']]
            X_test = self.test_processed[feature_cols]
            
            # Predict emotions and intensity
            self.test_emotions = self.emotion_model.predict(X_test)
            self.test_intensity = self.intensity_model.predict(X_test)
            
            # Get confidence scores
            emotion_proba = self.emotion_model.predict_proba(X_test)
            self.test_confidence = self.uncertainty_quantifier.calculate_confidence(emotion_proba)
            
            # Get uncertainty flags
            text_length = self.test_processed.get('text_length', pd.Series([50] * len(X_test))).values
            word_count = self.test_processed.get('word_count', pd.Series([10] * len(X_test))).values
            has_contradictions = self.test_processed.get('has_contradictions', pd.Series([0] * len(X_test))).values
            has_uncertainty = self.test_processed.get('has_uncertainty', pd.Series([0] * len(X_test))).values
            
            self.test_uncertain = self.uncertainty_quantifier.calculate_uncertainty_flag(
                self.test_confidence, text_length, word_count, has_contradictions, has_uncertainty
            )
            
            print(f"✓ Predictions generated")
            print(f"  - Average confidence: {self.test_confidence.mean():.3f}")
            print(f"  - Uncertain flags: {self.test_uncertain.sum()} / {len(X_test)}")
            
        except Exception as e:
            self.logger.log(f"Error making predictions: {e}", "ERROR")
            raise
    
    def run_decision_engine(self):
        """Run decision engine on predictions."""
        self.print_header("RUNNING DECISION ENGINE")
        
        try:
            # Create prediction dataframe
            pred_df = self.test_data.copy()
            pred_df['emotional_state'] = self.test_emotions
            pred_df['intensity'] = self.test_intensity
            
            # Run decision engine
            decisions = self.decision_engine.decide(pred_df)
            
            self.test_decisions = decisions
            print(f"✓ Decisions generated")
            
        except Exception as e:
            self.logger.log(f"Error running decision engine: {e}", "ERROR")
            raise
    
    def create_predictions_csv(self):
        """Create final predictions CSV."""
        self.print_header("CREATING PREDICTIONS CSV")
        
        try:
            predictions_df = pd.DataFrame({
                'id': self.test_data.get('id', range(len(self.test_data))),
                'predicted_state': self.test_emotions,
                'predicted_intensity': self.test_intensity,
                'confidence': self.test_confidence,
                'uncertain_flag': self.test_uncertain,
                'what_to_do': self.test_decisions['what_to_do'].values,
                'when_to_do': self.test_decisions['when_to_do'].values,
                'supportive_message': self.test_decisions['supportive_message'].values,
            })
            
            FileManager.save_df(predictions_df, 'results/predictions.csv')
            print(f"✓ Predictions saved: {predictions_df.shape}")
            
        except Exception as e:
            self.logger.log(f"Error creating predictions CSV: {e}", "ERROR")
            raise
    
    def evaluate_models(self):
        """Evaluate models on training data."""
        self.print_header("MODEL EVALUATION (Training Data)")
        
        try:
            # Prepare training features
            feature_cols = [col for col in self.train_processed.columns 
                           if col not in ['id', 'emotional_state', 'intensity', 'journal_text']]
            X_train = self.train_processed[feature_cols]
            
            # Emotion accuracy
            train_emotion_pred = self.emotion_model.predict(X_train)
            emotion_accuracy = self.metrics.calculate_accuracy(
                self.train_processed['emotional_state'], train_emotion_pred
            )
            print(f"✓ Emotion Accuracy: {emotion_accuracy:.3f}")
            
            # Intensity accuracy
            train_intensity_pred = self.intensity_model.predict(X_train)
            intensity_accuracy = self.metrics.calculate_accuracy(
                self.train_processed['intensity'], train_intensity_pred
            )
            print(f"✓ Intensity Accuracy: {intensity_accuracy:.3f}")
            
        except Exception as e:
            self.logger.log(f"Error evaluating models: {e}", "ERROR")
            raise
    
    def save_models(self):
        """Save trained models."""
        self.print_header("SAVING MODELS")
        
        try:
            FileManager.save_model(self.emotion_model, 'models/emotion_model.pkl')
            FileManager.save_model(self.intensity_model, 'models/intensity_model.pkl')
            FileManager.save_model(self.preprocessor, 'models/preprocessor.pkl')
            
        except Exception as e:
            self.logger.log(f"Error saving models: {e}", "ERROR")
            raise
    
    def run(self, train_path: str = 'data/training_data.csv', 
            test_path: str = 'data/test_data.csv'):
        """Run complete pipeline."""
        self.print_header("ARVYAX EMOTION DETECTION PIPELINE STARTED")
        
        try:
            self.load_data(train_path, test_path)
            self.preprocess_data()
            self.train_models()
            self.make_predictions()
            self.run_decision_engine()
            self.create_predictions_csv()
            self.evaluate_models()
            self.save_models()
            
            self.print_header("PIPELINE COMPLETED SUCCESSFULLY!")
            print("📁 Results saved in 'results/' folder:")
            print("   - predictions.csv: Test set predictions")
            print("   - pipeline.log: Execution log\n")
            
        except Exception as e:
            self.logger.log(f"Pipeline failed: {e}", "ERROR")
            raise


if __name__ == "__main__":
    pipeline = EmotionDetectionPipeline()
    pipeline.run()