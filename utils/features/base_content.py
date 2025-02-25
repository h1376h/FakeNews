from .base import FeatureExtractor
import pandas as pd
import numpy as np
from textblob import TextBlob
import nltk
from typing import List, Dict, Any, Set
import ssl
import re
import emoji
from abc import abstractmethod
import os

class BaseContentFeatureExtractor(FeatureExtractor):
    """Base class for content-based feature extractors with enhanced validation and computation.
    
    Supports two feature sets:
    1. Paper features (15 features):
        - content_polarity: Average sentiment polarity
        - content_subjectivity: Average subjectivity score
        - content_disagreement: Amount of tweets expressing disagreement
        - content_num_question: Number of tweets containing question marks
        - content_ratio_question: Ratio of tweets with question marks
        - content_num_exclamation: Number of tweets containing exclamation marks
        - content_ratio_exclamation: Ratio of tweets with exclamation marks
        - content_num_first_person: Number of tweets containing first-person pronouns
        - content_ratio_first_person: Ratio of tweets with first-person pronouns
        - content_num_second_person: Number of tweets containing second-person pronouns
        - content_ratio_second_person: Ratio of tweets with second-person pronouns
        - content_num_third_person: Number of tweets containing third-person pronouns
        - content_ratio_third_person: Ratio of tweets with third-person pronouns
        - content_num_smiley: Number of tweets containing smileys
        - content_ratio_smiley: Ratio of tweets with smileys
        
    2. Additional features (8 features):
        - content_num_info_request: Number of tweets requesting information
        - content_ratio_info_request: Ratio of tweets requesting information
        - content_num_support: Number of tweets supporting source tweet
        - content_ratio_support: Ratio of tweets supporting source tweet
        - content_num_disagreement: Number of tweets expressing disagreement
        - content_ratio_disagreement: Ratio of tweets expressing disagreement
        - content_num_polarity: Number of tweets containing polarity
        - content_num_subjectivity: Number of tweets containing subjectivity
    """
    
    # Core feature sets
    PAPER_FEATURES = {
        'content_polarity',
        'content_subjectivity',
        'content_disagreement',
        'content_num_question',
        'content_ratio_question',
        'content_num_exclamation',
        'content_ratio_exclamation',
        'content_num_first_person',
        'content_ratio_first_person',
        'content_num_second_person',
        'content_ratio_second_person',
        'content_num_third_person',
        'content_ratio_third_person',
        'content_num_smiley',
        'content_ratio_smiley'
    }
    
    ADDITIONAL_FEATURES = {
        'content_num_info_request',
        'content_ratio_info_request',
        'content_num_support',
        'content_ratio_support',
        'content_num_disagreement',
        'content_ratio_disagreement',
        'content_num_polarity',
        'content_num_subjectivity'
    }

    # Base patterns that can be extended by dataset-specific extractors
    BASE_PATTERNS = {
        'disagreement_words': [
            'false', 'fake', 'wrong', 'incorrect', 'lie',
            'hoax', 'debunk', 'conspiracy', 'misleading', 'untrue',
            'doubt', 'disputed', 'deny', 'denies', 'denied',
            'fabricated', 'baseless', 'unfounded', 'disinformation',
            'misinformation', 'propaganda', 'manipulated', 'deceptive',
            'inaccurate', 'misrepresented', 'unsubstantiated', 'dubious',
            'questionable', 'unproven', 'distorted', 'misquoted',
            'misattributed', 'unverified', 'rumor', 'alleged'
        ],
        'support_words': [
            'true', 'real', 'correct', 'accurate', 'right',
            'confirm', 'verified', 'legitimate', 'factual',
            'evidence', 'proof', 'proven', 'valid', 'authentic',
            'reliable', 'trustworthy', 'credible', 'substantiated',
            'corroborated', 'validated', 'fact-checked', 'confirmed',
            'documented', 'sourced', 'verified by', 'according to',
            'official statement', 'expert analysis', 'primary source'
        ],
        'negation_patterns': [
            r'\bnot\b',
            r"n't\b",
            r'\bno\b',
            r'\bnever\b',
            r'\bnor\b',
            r'\bwithout\b',
            r'\bdidn\'t\b',
            r'\bwasn\'t\b',
            r'\baren\'t\b',
            r'\bweren\'t\b',
            r'\bhasn\'t\b',
            r'\bhaven\'t\b',
            r'\bhadn\'t\b',
            r'\bwon\'t\b',
            r'\bwouldn\'t\b',
            r'\bshouldn\'t\b',
            r'\bcan\'t\b',
            r'\bcannot\b',
            r'\bcouldn\'t\b',
            r'\bisn\'t\b',
            r'\bdoesn\'t\b'
        ],
        'info_request_patterns': [
            r'\b(what|when|where|who|why|how)\b',
            r'\bis it true\b',
            r'\bconfirm\b',
            r'\bverify\b',
            r'\bprove\b',
            r'\bsource\b',
            r'\bevidence\b',
            r'\bany proof\b',
            r'\bcan you confirm\b',
            r'\bplease verify\b',
            r'\bsource\?+\b',
            r'\bcitation needed\b',
            r'\breference\?+\b',
            r'\bmore info\b',
            r'\bany updates?\b',
            r'\bconfirmation\?+\b'
        ],
        'smiley_patterns': [
            r'[:=]-?\)',  # :) =)
            r'[:=]-?D',   # :D =D
            r'[:=]-?\}',  # :} =}
            r'[:=]-?]',   # :] =]
            r'[:=]-?p',   # :p =p
            r'[:=]-?P',   # :P =P
            r';-?\)',     # ;)
            r'\([:=]-?\)', # (:) (=)
            r'[:=]3',     # :3
            r'<3',        # heart
            r'♥',         # heart symbol
            r'☺',         # smiling face
            r'😊|😃|😄|😁|😆|😅|😂|🤣|☺️|😊|😇|🙂|🙃|😉|😌|😍|🥰|😘|😗|😙|😚|��|😛|😝|😜|🤪|🤨|🧐|🤓|😎|🤩|🥳'  # emojis
        ],
        'pronoun_patterns': {
            'first_person': r'\b(i|me|my|mine|we|us|our|ours|myself|ourselves)\b',
            'second_person': r'\b(you|your|yours|yourself|yourselves)\b',
            'third_person': r'\b(he|him|his|she|her|hers|it|its|they|them|their|theirs|himself|herself|itself|themselves)\b'
        }
    }
    
    @classmethod
    def _setup_nltk(cls):
        """Set up NLTK resources safely with improved error handling and resource verification."""
        try:
            # SSL setup for NLTK downloads
            try:
                _create_unverified_https_context = ssl._create_unverified_context
            except AttributeError:
                pass
            else:
                ssl._create_default_https_context = _create_unverified_https_context
            
            # Resources we need
            resources = ['punkt', 'averaged_perceptron_tagger']
            nltk_data_path = nltk.data.path

            # Check if the resources are already available
            missing_resources = []
            for resource in resources:
                try:
                    nltk.data.find(f'tokenizers/{resource}')
                except LookupError:
                    missing_resources.append(resource)
            
            # Only download missing resources
            for resource in missing_resources:
                print(f"Downloading NLTK resource: {resource}")
                nltk.download(resource, quiet=True)

            # Verify downloads were successful
            for resource in resources:
                try:
                    nltk.data.find(f'tokenizers/{resource}')
                except LookupError:
                    print(f"Warning: NLTK resource {resource} not available despite download attempt.")
                    # Create a fallback folder if needed
                    nltk_custom_path = os.path.join(os.getcwd(), 'nltk_data')
                    if nltk_custom_path not in nltk_data_path:
                        nltk.data.path.append(nltk_custom_path)
                    # Try download one more time to custom path
                    nltk.download(resource, download_dir=nltk_custom_path, quiet=True)
        except Exception as e:
            print(f"Warning: Could not download NLTK data: {str(e)}")
            print("Text processing features may not work correctly.")
    
    def __init__(self, df: pd.DataFrame, include_additional_features: bool = False):
        """Initialize the feature extractor.
        
        Args:
            df: Input DataFrame
            include_additional_features: Whether to include additional features not in the paper
        """
        super().__init__(df)
        self.include_additional_features = include_additional_features
        self.features_to_extract = self.PAPER_FEATURES.copy()
        if include_additional_features:
            self.features_to_extract.update(self.ADDITIONAL_FEATURES)
        
        # Initialize patterns with base patterns
        self.patterns = self.BASE_PATTERNS.copy()
        # Allow dataset-specific patterns to be added
        self.extend_patterns()
        
        # Set up NLTK only once
        self._setup_nltk()
    
    def _safe_extend_patterns(self, pattern_key: str, new_patterns: list) -> None:
        """Safely extend pattern list without duplicates.
        
        Args:
            pattern_key: Key of the pattern list to extend
            new_patterns: New patterns to add
        """
        if pattern_key not in self.patterns:
            return
            
        # Convert existing patterns to set for O(1) lookup
        existing = set(self.patterns[pattern_key])
        # Only add patterns that don't already exist
        unique_new = [p for p in new_patterns if p not in existing]
        self.patterns[pattern_key].extend(unique_new)

    def extend_patterns(self):
        """Override this method to add dataset-specific patterns."""
        pass
    
    def _initialize_feature_columns(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Initialize feature columns with NaN values."""
        for col in columns:
            if col not in df.columns:
                df[col] = np.nan
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Handle missing values in feature columns."""
        for col in columns:
            if col in df.columns:
                df.loc[:, col] = df[col].fillna(0)
        return df
    
    def _get_text_polarity(self, text: str) -> float:
        """Calculate the polarity score of text using TextBlob."""
        try:
            return TextBlob(str(text)).sentiment.polarity
        except Exception:
            return 0.0
    
    def _get_text_subjectivity(self, text: str) -> float:
        """Calculate the subjectivity score of text using TextBlob."""
        try:
            return TextBlob(str(text)).sentiment.subjectivity
        except Exception:
            return 0.0
    
    def _count_question_marks(self, text: str) -> int:
        """Count question marks in text."""
        return str(text).count('?')
    
    def _count_exclamation_points(self, text: str) -> int:
        """Count exclamation points in text."""
        return str(text).count('!')
    
    def _count_pronouns(self, text: str) -> Dict[str, int]:
        """Count pronouns in text by type.
        
        Returns:
            Dictionary with counts for first_person, second_person, and third_person pronouns
        """
        # Ensure text is not None
        text = str(text) if text is not None else ""
        
        pronouns = {
            'first_person': 0,
            'second_person': 0,
            'third_person': 0
        }
        
        try:
            # Try to use our pronoun pattern detection
            for pronoun_type, pattern in self.patterns['pronoun_patterns'].items():
                pronouns[pronoun_type] = len(re.findall(pattern, text.lower()))
        except Exception as e:
            # Fallback to basic word matching in case regex fails
            print(f"Warning: Error in pronoun detection: {str(e)}. Using fallback method.")
            words = text.lower().split()
            
            # Basic word lists for pronouns
            first_person = ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours', 'myself', 'ourselves']
            second_person = ['you', 'your', 'yours', 'yourself', 'yourselves']
            third_person = ['he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 
                           'they', 'them', 'their', 'theirs', 'himself', 'herself', 
                           'itself', 'themselves']
            
            pronouns['first_person'] = sum(word in first_person for word in words)
            pronouns['second_person'] = sum(word in second_person for word in words)
            pronouns['third_person'] = sum(word in third_person for word in words)
        
        return pronouns
    
    def _count_smileys(self, text: str) -> int:
        """Count smileys/emoticons in text using regex patterns.
        
        Returns:
            Integer count of smileys found
        """
        # Ensure text is not None
        text = str(text) if text is not None else ""
        
        try:
            # Try to use the smiley pattern to find matches
            return len(re.findall(self.patterns['smiley_pattern'], text))
        except Exception:
            # Fallback smiley detection
            return sum(1 for char in text if char in ':) :( ;) :D :P :p ^^'.split())
    
    def _check_negation(self, text: str, window_size: int = 5) -> bool:
        """Check for negation words in text within a window size of key terms.
        
        Args:
            text: Text to analyze
            window_size: Number of words to look for negation terms
            
        Returns:
            Boolean indicating whether negation is detected
        """
        # Ensure text is not None
        text = str(text) if text is not None else ""
        
        # Tokenize text into words for window analysis
        words = text.lower().split()
        
        # Find negation words
        for i, word in enumerate(words):
            # Check if current word contains any negation pattern
            if any(re.search(pattern, word) for pattern in self.patterns['negation_patterns']):
                # Look for support words in a window around the negation
                window_start = max(0, i - window_size)
                window_end = min(len(words), i + window_size + 1)
                window = words[window_start:window_end]
                
                # Check if any support word is in the window
                if any(support_word in window for support_word in self.patterns['support_words']):
                    return True
        
        return False

    def _check_disagreement(self, text: str) -> bool:
        """Check if text expresses disagreement or denial.
        
        Returns:
            Boolean indicating disagreement
        """
        # Ensure text is not None
        text = str(text) if text is not None else ""
        
        # Simple pattern matching for disagreement words
        text_lower = text.lower()
        return any(word in text_lower for word in self.patterns['disagreement_words'])

    def _check_support(self, text: str) -> bool:
        """Check if text expresses support or agreement.
        
        Returns:
            Boolean indicating support
        """
        # Ensure text is not None
        text = str(text) if text is not None else ""
        
        # Simple pattern matching for support words
        text_lower = text.lower()
        return any(word in text_lower for word in self.patterns['support_words'])

    def _check_info_request(self, text: str) -> bool:
        """Check if text contains an information request.
        
        Returns:
            Boolean indicating an information request
        """
        # Ensure text is not None
        text = str(text) if text is not None else ""
        
        # Check for question marks or question phrases
        return '?' in text or any(pattern in text.lower() for pattern in self.patterns['info_request_patterns'])
    
    def _process_tweets(self, tweets: List[str]) -> Dict[str, float]:
        """Process a list of tweets and extract all content features.
        
        Args:
            tweets: List of tweet texts to process
            
        Returns:
            Dictionary containing content features based on configuration
        """
        num_tweets = len(tweets)
        if num_tweets == 0:
            return {}
            
        try:
            # Calculate all metrics
            polarities = [self._get_text_polarity(tweet) for tweet in tweets]
            subjectivities = [self._get_text_subjectivity(tweet) for tweet in tweets]
            question_marks = [self._count_question_marks(tweet) for tweet in tweets]
            exclamation_points = [self._count_exclamation_points(tweet) for tweet in tweets]
            pronoun_counts = [self._count_pronouns(tweet) for tweet in tweets]
            smiley_counts = [self._count_smileys(tweet) for tweet in tweets]
            disagreements = [self._check_disagreement(text) for text in tweets]
            supports = [self._check_support(text) for text in tweets]
            info_requests = [self._check_info_request(text) for text in tweets]
            
            # Initialize features dictionary with paper features
            features = {
                'content_polarity': np.mean(polarities),
                'content_subjectivity': np.mean(subjectivities),
                'content_disagreement': sum(disagreements),
                'content_num_question': sum(question_marks),
                'content_ratio_question': sum(1 for count in question_marks if count > 0) / num_tweets if num_tweets > 0 else 0,
                'content_num_exclamation': sum(exclamation_points),
                'content_ratio_exclamation': sum(1 for count in exclamation_points if count > 0) / num_tweets if num_tweets > 0 else 0,
                'content_num_first_person': sum(count['first_person'] for count in pronoun_counts),
                'content_ratio_first_person': sum(1 for count in pronoun_counts if count['first_person'] > 0) / num_tweets if num_tweets > 0 else 0,
                'content_num_second_person': sum(count['second_person'] for count in pronoun_counts),
                'content_ratio_second_person': sum(1 for count in pronoun_counts if count['second_person'] > 0) / num_tweets if num_tweets > 0 else 0,
                'content_num_third_person': sum(count['third_person'] for count in pronoun_counts),
                'content_ratio_third_person': sum(1 for count in pronoun_counts if count['third_person'] > 0) / num_tweets if num_tweets > 0 else 0,
                'content_num_smiley': sum(smiley_counts),
                'content_ratio_smiley': sum(1 for count in smiley_counts if count > 0) / num_tweets if num_tweets > 0 else 0
            }
            
            # Add additional features if configured
            if self.include_additional_features:
                additional_features = {
                    'content_num_info_request': sum(info_requests),
                    'content_ratio_info_request': sum(info_requests) / num_tweets if num_tweets > 0 else 0,
                    'content_num_support': sum(supports),
                    'content_ratio_support': sum(supports) / num_tweets if num_tweets > 0 else 0,
                    'content_num_disagreement': sum(disagreements),
                    'content_ratio_disagreement': sum(disagreements) / num_tweets if num_tweets > 0 else 0,
                    'content_num_polarity': sum(1 for p in polarities if abs(p) > 0.1),
                    'content_num_subjectivity': sum(1 for s in subjectivities if s > 0.1)
                }
                features.update(additional_features)
            
            # Validate that all required features are present
            missing_features = self.features_to_extract - set(features.keys())
            if missing_features:
                raise ValueError(f"Missing required content features: {missing_features}")
                
            return features
            
        except Exception as e:
            raise ValueError(f"Error processing content features: {str(e)}")
    
    @abstractmethod
    def extract_features(self) -> pd.DataFrame:
        """Extract content features from the dataset.
        
        This method must be implemented by dataset-specific extractors to handle
        the unique structure of each dataset (e.g., how to get source tweets and reactions).
        """
        pass