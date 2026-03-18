"""
Machine learning models for emotion detection and intensity prediction.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')


class EmotionalStateModel:
    """Multi-class classifier for emotional state prediction."""
    
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.class_names = None
    
    def build_ensemble(self):
        """Build ensemble model combining multiple algorithms."""
        lr = LogisticRegression(max_iter=1000, random_state=42)
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        svc = SVC(kernel='rbf', probability=True, random_state=42)
        xgb = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='mlogloss', verbose=0)
        
        self.model = VotingClassifier(
            estimators=[('lr', lr), ('rf', rf), ('svc', svc), ('xgb', xgb)],
            voting='soft'
        )
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Train the model."""
        self.feature_names = X.columns.tolist()
        self.class_names = self.label_encoder.fit_transform(y)
        
        self.build_ensemble()
        self.model.fit(X, self.class_names)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        predictions = self.model.predict(X)
        return self.label_encoder.inverse_transform(predictions)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        return self.model.predict_proba(X)
    
    def get_confidence(self, X: pd.DataFrame) -> np.ndarray:
        """Get confidence scores (max probability)."""
        proba = self.predict_proba(X)
        return np.max(proba, axis=1)
    
    def get_feature_importance(self):
        """Extract feature importance from Random Forest component."""
        rf_model = self.model.named_estimators_['rf']
        importance = rf_model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance_df


class IntensityModel:
    """Ordinal classifier for intensity prediction (1-5 scale)."""
    
    def __init__(self):
        self.model = None
        self.feature_names = None
    
    def build_model(self):
        """Build XGBoost model for ordinal regression."""
        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss',
            scale_pos_weight=1,
            verbose=0
        )
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Train the model."""
        self.feature_names = X.columns.tolist()
        self.build_model()
        self.model.fit(X, y)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions (1-5)."""
        predictions = self.model.predict(X)
        predictions = np.clip(predictions, 1, 5)
        return predictions.astype(int)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get probability distribution over intensity levels."""
        return self.model.predict_proba(X)
    
    def get_feature_importance(self):
        """Get feature importance scores."""
        importance = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance_df