from .base_content import BaseContentFeatureExtractor
import pandas as pd
from typing import Dict, Any

class BuzzFeedContentFeatureExtractor(BaseContentFeatureExtractor):
    """Features based on the content of tweets in BuzzFeed dataset."""
    
    def __init__(self, df: pd.DataFrame, include_additional_features: bool = False):
        """Initialize the BuzzFeed content feature extractor.
        
        Args:
            df: Input DataFrame
            include_additional_features: Whether to include additional features not in the paper
        """
        super().__init__(df, include_additional_features)
    
    def extend_patterns(self):
        """
        Extend the patterns with BuzzFeed-specific patterns.
        Currently unimplemented.
        """
        pass
    
    def extract_features(self) -> pd.DataFrame:
        """Extract content features from the BuzzFeed dataset."""
        df = self.df.copy()
        
        # Initialize feature columns with NaN
        df = self._initialize_feature_columns(df, list(self.features_to_extract))
        
        # Process each article and its reactions
        for idx, row in df.iterrows():
            # Get all tweets in the thread
            all_tweets = [row['mainText']] + (row.get('reaction_texts', []) or [])
            
            # Process tweets and store features
            features = self._process_tweets(all_tweets)
            for col, value in features.items():
                df.at[idx, col] = value
        
        # Handle missing values
        df = self._handle_missing_values(df, list(self.features_to_extract))
        
        return df