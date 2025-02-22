from .base import FeatureExtractor
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Set
from collections import defaultdict

class BaseStructuralFeatureExtractor(FeatureExtractor):
    """Base class for structural feature extractors with enhanced validation and computation.
    
    Supports exactly 14 structural features:
    1. structural_num_tweets_with_mentions (int64): Number of tweets containing mentions
    2. structural_ratio_tweets_with_hashtags (float64): Ratio of tweets containing hashtags
    3. structural_conversation_depth (int64): Depth of conversation tree
    4. structural_num_tweets_with_hashtags (int64): Number of tweets containing hashtags
    5. structural_thread_lifetime_minutes (float64): Time between first and last tweet
    6. structural_num_tweets_with_urls (int64): Number of tweets containing URLs
    7. structural_num_tweets (int64): Total number of tweets in thread
    8. structural_num_tweets_with_media (int64): Number of tweets with images/video
    9. structural_ratio_tweets_with_urls (float64): Ratio of tweets containing URLs
    10. structural_avg_tweet_length (float64): Average length of tweets
    11. structural_ratio_tweets_with_media (float64): Ratio of tweets containing media
    12. structural_ratio_tweets_with_mentions (float64): Ratio of tweets containing mentions
    13. structural_num_retweets (int64): Number of retweets
    14. structural_ratio_retweets (float64): Ratio of retweets to total tweets
    
    Attributes:
        STRUCTURAL_FEATURES: Set of all supported feature names
        URL_PATTERNS: Common URL patterns to detect links
        MEDIA_PATTERNS: Common media URL and indicator patterns
    """
    
    STRUCTURAL_FEATURES = {
        'structural_num_tweets_with_mentions',
        'structural_ratio_tweets_with_hashtags',
        'structural_conversation_depth',
        'structural_num_tweets_with_hashtags',
        'structural_thread_lifetime_minutes',
        'structural_num_tweets_with_urls',
        'structural_num_tweets',
        'structural_num_tweets_with_media',
        'structural_ratio_tweets_with_urls',
        'structural_avg_tweet_length',
        'structural_ratio_tweets_with_media',
        'structural_ratio_tweets_with_mentions',
        'structural_num_retweets',
        'structural_ratio_retweets'
    }
    
    URL_PATTERNS = [
        'http://', 'https://', 'www.', '.com', '.org', '.edu', '.gov',
        '.net', '.mil', '.int', '.io', '.ai', '.app', '.dev', '.co',
        '.info', '.biz', '.me', '.tv', '.blog', '.xyz', '.tech', '.online',
        '.site', '.web', '.cloud', '.digital', '.link', '.click', '.space',
        '.store', '.shop', '.academy', '.agency', '.news', '.media', '.live',
        '.social', '.wiki', '.uk', '.us', '.eu', '.ca', '.au', '.de', '.fr',
        '.jp', '.cn', '.ru', '.in', '.br', '.es', '.it', '.nl', '.pl'
    ]
    
    MEDIA_PATTERNS = [
        'pic.twitter.com', 'instagram.com', 'youtube.com', 'twitpic.com',
        'vine.co', 'photo:', 'video:', '.jpg', '.jpeg', '.png', '.gif', '.mp4',
        'vimeo.com', 'flickr.com', 'imgur.com', 'giphy.com', 'gfycat.com',
        'streamable.com', 'twitch.tv', 'dailymotion.com', '.webp', '.svg',
        '.mov', '.avi', '.wmv', '.flv', '.mkv', '.webm', 'photos.app.goo.gl',
        'media.giphy.com', 'tenor.com', '.heic', '.tiff', '.bmp', 'facebook.com/video',
        'snapchat.com', 'tiktok.com', '.m4v', '.3gp', 'photos.google.com',
        'drive.google.com/file', 'dropbox.com', 'onedrive.live.com', 'icloud.com/photos'
    ]
    
    def _calculate_avg_tweet_length(self, texts: List[str]) -> float:
        """Calculate average length of tweets in a thread.
        
        Args:
            texts: List of tweet texts
            
        Returns:
            float: Average length of tweets
        """
        if not texts:
            return 0.0
        try:
            # Convert to strings and filter out None values
            valid_texts = [str(text) for text in texts if text is not None]
            if not valid_texts:
                return 0.0
            return np.mean([len(text) for text in valid_texts])
        except Exception:
            return 0.0
    
    def _count_tweets_with_hashtags(self, texts: List[str]) -> int:
        """Count tweets containing hashtags.
        
        Args:
            texts: List of tweet texts
            
        Returns:
            int: Number of tweets containing hashtags
        """
        if not texts:
            return 0
        try:
            return sum(1 for text in texts if text is not None and '#' in str(text))
        except Exception:
            return 0
    
    def _count_tweets_with_media(self, texts: List[str]) -> int:
        """Count tweets containing media (images or video).
        
        Args:
            texts: List of tweet texts
            
        Returns:
            int: Number of tweets containing media
        """
        if not texts:
            return 0
        try:
            return sum(
                1 for text in texts
                if text is not None and 
                any(indicator in str(text).lower() for indicator in self.MEDIA_PATTERNS)
            )
        except Exception:
            return 0
    
    def _count_tweets_with_mentions(self, texts: List[str]) -> int:
        """Count tweets containing mentions.
        
        Args:
            texts: List of tweet texts
            
        Returns:
            int: Number of tweets containing mentions
        """
        if not texts:
            return 0
        try:
            return sum(1 for text in texts if text is not None and '@' in str(text))
        except Exception:
            return 0
    
    def _count_retweets(self, texts: List[str]) -> int:
        """Count number of retweets in the thread.
        
        Args:
            texts: List of tweet texts
            
        Returns:
            int: Number of retweets
        """
        if not texts:
            return 0
        try:
            return sum(1 for text in texts if text is not None and str(text).startswith('RT @'))
        except Exception:
            return 0
    
    def _count_tweets_with_urls(self, texts: List[str]) -> int:
        """Count tweets containing URLs.
        
        Args:
            texts: List of tweet texts
            
        Returns:
            int: Number of tweets containing URLs
        """
        if not texts:
            return 0
        try:
            return sum(
                1 for text in texts
                if text is not None and 
                any(pattern in str(text).lower() for pattern in self.URL_PATTERNS)
            )
        except Exception:
            return 0
    
    def _calculate_thread_lifetime(self, timestamps: List[datetime]) -> float:
        """Calculate thread lifetime in minutes.
        
        Args:
            timestamps: List of tweet timestamps
            
        Returns:
            float: Thread lifetime in minutes
        """
        if not timestamps:
            return 0.0
        
        try:
            # Filter out None values
            valid_timestamps = [t for t in timestamps if t is not None]
            if not valid_timestamps:
                return 0.0
                
            # Calculate time difference in minutes
            time_diff = (max(valid_timestamps) - min(valid_timestamps)).total_seconds() / 60
            return max(0.0, time_diff)
        except (ValueError, TypeError):
            return 0.0
    
    def _calculate_conversation_depth(self, tweet_ids: List[str], reply_to_ids: List[str]) -> int:
        """Calculate the depth of the conversation tree by analyzing reply chains.
        
        Args:
            tweet_ids: List of tweet IDs
            reply_to_ids: List of IDs that each tweet is replying to
            
        Returns:
            int: The maximum depth of the conversation tree. A depth of 1 means only
                 the source tweet exists, 2 means direct replies exist, and higher values
                 indicate nested reply chains.
        """
        if not tweet_ids or not reply_to_ids:
            return 1  # Source tweet only
            
        # Create a mapping of tweet IDs to their depths using defaultdict
        tweet_depths = defaultdict(lambda: 0)
        tweet_depths[str(tweet_ids[0])] = 1  # Source tweet has depth 1
        max_depth = 1
        
        try:
            # Create a mapping of replies for faster lookup
            reply_map = defaultdict(list)
            for tweet_id, reply_to_id in zip(tweet_ids[1:], reply_to_ids[1:]):
                if tweet_id and reply_to_id:
                    reply_map[str(reply_to_id)].append(str(tweet_id))
            
            # Process tweets level by level using BFS
            current_level = [str(tweet_ids[0])]  # Start with source tweet
            while current_level:
                next_level = []
                for tweet_id in current_level:
                    current_depth = tweet_depths[tweet_id]
                    # Add all replies to this tweet to the next level
                    for reply_id in reply_map[tweet_id]:
                        tweet_depths[reply_id] = current_depth + 1
                        next_level.append(reply_id)
                        max_depth = max(max_depth, current_depth + 1)
                current_level = next_level
                
        except Exception as e:
            # Log error but continue processing
            print(f"Error calculating conversation depth: {str(e)}")
        
        return max_depth
    
    def _process_structural_features(self, texts: List[str], timestamps: List[datetime] = None,
                                   tweet_ids: List[str] = None, reply_to_ids: List[str] = None) -> Dict[str, float]:
        """Process a list of tweets and extract all structural features.
        
        Args:
            texts: List of tweet texts to process
            timestamps: Optional list of tweet timestamps
            tweet_ids: Optional list of tweet IDs
            reply_to_ids: Optional list of IDs that each tweet is replying to
            
        Returns:
            Dictionary containing exactly 14 structural features
            
        Raises:
            ValueError: If any required features are missing
        """
        num_tweets = len(texts)
        if num_tweets == 0:
            return {}
        
        try:
            # Calculate basic metrics
            tweets_with_hashtags = self._count_tweets_with_hashtags(texts)
            tweets_with_media = self._count_tweets_with_media(texts)
            tweets_with_mentions = self._count_tweets_with_mentions(texts)
            retweets = self._count_retweets(texts)
            tweets_with_urls = self._count_tweets_with_urls(texts)
            
            # Calculate all 14 required structural features
            features = {
                'structural_num_tweets': num_tweets,
                'structural_num_tweets_with_mentions': tweets_with_mentions,
                'structural_ratio_tweets_with_mentions': tweets_with_mentions / num_tweets if num_tweets > 0 else 0,
                'structural_num_tweets_with_hashtags': tweets_with_hashtags,
                'structural_ratio_tweets_with_hashtags': tweets_with_hashtags / num_tweets if num_tweets > 0 else 0,
                'structural_num_tweets_with_urls': tweets_with_urls,
                'structural_ratio_tweets_with_urls': tweets_with_urls / num_tweets if num_tweets > 0 else 0,
                'structural_num_tweets_with_media': tweets_with_media,
                'structural_ratio_tweets_with_media': tweets_with_media / num_tweets if num_tweets > 0 else 0,
                'structural_num_retweets': retweets,
                'structural_ratio_retweets': retweets / num_tweets if num_tweets > 0 else 0,
                'structural_avg_tweet_length': self._calculate_avg_tweet_length(texts),
                'structural_thread_lifetime_minutes': self._calculate_thread_lifetime(timestamps) if timestamps else 0.0,
                'structural_conversation_depth': self._calculate_conversation_depth(tweet_ids, reply_to_ids) if tweet_ids and reply_to_ids else 1
            }
            
            # Validate that all required features are present
            missing_features = self.STRUCTURAL_FEATURES - set(features.keys())
            if missing_features:
                raise ValueError(f"Missing required structural features: {missing_features}")
                
            return features
            
        except Exception as e:
            raise ValueError(f"Error processing structural features: {str(e)}")
    
    def extract_features(self) -> pd.DataFrame:
        """Extract structural features from the dataset.
        
        Returns:
            DataFrame with exactly 14 structural features:
            1. structural_num_tweets_with_mentions (int64)
            2. structural_ratio_tweets_with_hashtags (float64)
            3. structural_conversation_depth (int64)
            4. structural_num_tweets_with_hashtags (int64)
            5. structural_thread_lifetime_minutes (float64)
            6. structural_num_tweets_with_urls (int64)
            7. structural_num_tweets (int64)
            8. structural_num_tweets_with_media (int64)
            9. structural_ratio_tweets_with_urls (float64)
            10. structural_avg_tweet_length (float64)
            11. structural_ratio_tweets_with_media (float64)
            12. structural_ratio_tweets_with_mentions (float64)
            13. structural_num_retweets (int64)
            14. structural_ratio_retweets (float64)
            
        Raises:
            NotImplementedError: If not implemented by subclass
            ValueError: If feature validation fails
        """
        raise NotImplementedError("Subclasses must implement extract_features method")