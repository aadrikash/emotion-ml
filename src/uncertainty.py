"""
Uncertainty quantification for predictions.
"""
import numpy as np
import pandas as pd


class UncertaintyQuantifier:
    """Assess confidence and uncertainty in predictions."""
    
    def __init__(self, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold
    
    def calculate_confidence(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Calculate confidence as max probability.
        Range: [0, 1] where 1 is most confident.
        """
        return np.max(probabilities, axis=1)
    
    def calculate_uncertainty_flag(self, 
                                   confidence: np.ndarray,
                                   text_length: np.ndarray,
                                   word_count: np.ndarray,
                                   has_contradictions: np.ndarray,
                                   has_uncertainty: np.ndarray) -> np.ndarray:
        """
        Generate uncertainty flag (1 = uncertain, 0 = confident).
        
        Uncertainty factors:
        - Low confidence score
        - Very short text
        - Text contains contradictions
        - Text contains uncertainty markers
        """
        uncertain = np.zeros(len(confidence))
        
        # Low confidence
        uncertain[confidence < self.confidence_threshold] = 1
        
        # Very short text (< 5 words)
        uncertain[word_count < 5] = 1
        
        # Contains contradictions
        uncertain[has_contradictions > 0.5] = 1
        
        # Contains uncertainty markers
        uncertain[has_uncertainty > 0.5] = 1
        
        return uncertain.astype(int)
    
    def calculate_entropy(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Calculate entropy of predictions.
        Higher entropy = more uncertain.
        """
        # Avoid log(0)
        probabilities = np.clip(probabilities, 1e-10, 1)
        entropy = -np.sum(probabilities * np.log(probabilities), axis=1)
        return entropy
    
    def get_uncertainty_details(self, df: pd.DataFrame, 
                               confidence: np.ndarray,
                               uncertainty_flag: np.ndarray) -> pd.DataFrame:
        """Create detailed uncertainty report."""
        return pd.DataFrame({
            'confidence': confidence,
            'uncertain_flag': uncertainty_flag,
            'text_is_short': (df.get('word_count', pd.Series([0] * len(df))) < 5).astype(int),
            'has_contradictions': df.get('has_contradictions', pd.Series([0] * len(df))).astype(int),
            'has_uncertainty_markers': df.get('has_uncertainty', pd.Series([0] * len(df))).astype(int),
        })