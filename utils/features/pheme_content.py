from .base_content import BaseContentFeatureExtractor
import pandas as pd
from textblob import TextBlob
from nltk.tokenize import word_tokenize
import nltk
from typing import List, Dict, Any
import ssl
import os
import re
import emoji
import numpy as np

"""Content-based features for PHEME dataset."""
class PhemeContentFeatureExtractor(BaseContentFeatureExtractor):
    """Features based on the content of tweets in PHEME dataset."""
    
    def __init__(self, df: pd.DataFrame, include_additional_features: bool = False):
        """Initialize the PHEME content feature extractor.
        
        Args:
            df: Input DataFrame
            include_additional_features: Whether to include additional features not in the paper
        """
        super().__init__(df, include_additional_features)
        # Download required NLTK data
        try:
            # Handle SSL certificate issues
            try:
                _create_unverified_https_context = ssl._create_unverified_context
            except AttributeError:
                pass
            else:
                ssl._create_default_https_context = _create_unverified_https_context
            
            # Download required NLTK data
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
        except Exception as e:
            print(f"Warning: Could not download NLTK data: {str(e)}")
    
    def extend_patterns(self):
        """
        Extend the patterns with PHEME-specific patterns.
        Currently unimplemented.
        """
        pass
    
    def extract_features(self) -> pd.DataFrame:
        """Extract content features from the PHEME dataset."""
        df = self.df.copy()
        
        # Initialize feature columns with NaN
        df = self._initialize_feature_columns(df, list(self.features_to_extract))
        
        # Process source tweet and reactions
        for row_idx, row in df.iterrows():
            # Get all tweets in the thread
            all_tweets = [row['source_tweet_text']] + row.get('reaction_texts', [])
            
            # Process tweets and store features
            features = self._process_tweets(all_tweets)
            for col, value in features.items():
                df.at[row_idx, col] = value
        
        # Handle missing values
        df = self._handle_missing_values(df, list(self.features_to_extract))
        
        return df 