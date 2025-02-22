from .base_content import BaseContentFeatureExtractor
import pandas as pd
from typing import Dict, Any

"""Content-based features for CREDBANK dataset."""
class CredbankContentFeatureExtractor(BaseContentFeatureExtractor):
    """Features based on the content of tweets in CREDBANK dataset."""
    
    def __init__(self, df: pd.DataFrame, include_additional_features: bool = False):
        """Initialize the CREDBANK content feature extractor.
        
        Args:
            df: Input DataFrame
            include_additional_features: Whether to include additional features not in the paper
        """
        super().__init__(df, include_additional_features)
    
    def extend_patterns(self):
        """
        Extend the patterns with CREDBANK-specific patterns.
        Currently unimplemented.
        """
        pass
    
    def extract_features(self) -> pd.DataFrame:
        """Extract content features from the CREDBANK dataset."""
        df = self.df.copy()
        
        # Initialize feature columns with NaN
        df = self._initialize_feature_columns(df, list(self.features_to_extract))
        
        # Process each row
        for idx, row in df.iterrows():
            # Use Reasons field as our text source
            texts = row.get('Reasons', [])
            if not isinstance(texts, list):
                texts = []
            
            # Process tweets and store features
            features = self._process_tweets(texts)
            for col, value in features.items():
                df.at[idx, col] = value
        
        # Handle missing values
        df = self._handle_missing_values(df, list(self.features_to_extract))
        
        return df 