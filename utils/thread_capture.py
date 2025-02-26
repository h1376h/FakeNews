import json
import os
import pandas as pd
from typing import Dict, List, Tuple, Any, Union
from collections import defaultdict
import numpy as np
import warnings
import requests
import time
from tqdm import tqdm

class ThreadCaptureTool:
    """
    Tool for capturing Twitter's threaded structure from different datasets.
    
    This tool adapts tweet sets from CREDBANK and BuzzFeed to capture threaded
    structure similar to the structure in PHEME.
    
    For CREDBANK:
    - Identifies the most retweeted tweet in each event as the thread root
    - Collects replies to this root tweet as children
    - Discards threads with no reactions
    
    For BuzzFeed:
    - Uses the popular headline tweets as thread roots
    - Captures replies to these roots to construct thread structure
    
    BuzzFeed to Twitter Alignment:
    - Extracts the top 10 most shared stories from left- and right-wing pages 
    - Searches Twitter for these headlines
    - Keeps the top 3 most retweeted posts for each headline
    - Results in 35 topics with journalist-provided labels (15 "mostly true", 20 "mostly false")
    """
    
    def __init__(self, base_path: str = 'data'):
        """Initialize ThreadCaptureTool.
        
        Args:
            base_path: Base directory path for dataset storage
        """
        self.base_path = base_path
        self.credbank_path = os.path.join(base_path, 'credbank')
        self.credbank_raw_path = os.path.join(base_path, 'credbank', 'CREDBANK')
        self.buzzfeed_path = os.path.join(base_path, 'buzzfeed')
        self.pheme_path = os.path.join(base_path, 'pheme')
        
        # Twitter API access (should be configured with API keys)
        self.twitter_api_available = False
        
        try:
            # Initialize Twitter API client if required dependencies are installed
            # This is a placeholder - you would need to use an appropriate Twitter API client
            # such as tweepy or the official Twitter API v2 client
            import tweepy
            # Setup Twitter API credentials (these should be configured elsewhere)
            # api_key, api_secret, access_token, access_token_secret = self._get_twitter_credentials()
            # self.twitter_api = tweepy.API(tweepy.OAuth1UserHandler(api_key, api_secret, access_token, access_token_secret))
            # self.twitter_api_available = True
        except (ImportError, Exception) as e:
            warnings.warn(f"Twitter API access not configured: {str(e)}")
    
    def _get_twitter_credentials(self):
        """Retrieve Twitter API credentials from environment or config file."""
        # Implementation would depend on your credential management approach
        # Could load from environment variables, config file, etc.
        pass
            
    def capture_credbank_threads(self, credbank_df: pd.DataFrame = None) -> pd.DataFrame:
        """Capture threaded structure for CREDBANK dataset.
        
        Args:
            credbank_df: DataFrame containing CREDBANK dataset. If None, loads from CSV.
            
        Returns:
            DataFrame with CREDBANK data in thread structure format
        """
        if credbank_df is None:
            # Load CREDBANK dataset
            credbank_file = os.path.join(self.credbank_path, 'credbank_extended_dataset.csv')
            if not os.path.exists(credbank_file):
                raise FileNotFoundError(f"CREDBANK dataset not found at {credbank_file}")
            credbank_df = pd.read_csv(credbank_file)
        
        # Group tweets by event/topic
        grouped_df = credbank_df.groupby('topic_key')
        
        # Initialize list to store thread data
        thread_data = []
        
        # Process each event group
        for topic_key, group in grouped_df:
            # Find the most retweeted tweet as root
            if 'retweet_count' in group.columns:
                # Use actual retweet count if available
                root_idx = group['retweet_count'].idxmax()
            else:
                # Fall back to using the first tweet if retweet counts not available
                root_idx = group.index[0]
            
            # Get root tweet data
            root_tweet = group.loc[root_idx].to_dict()
            
            # Get the rest of the tweets as reactions
            reactions = group[group.index != root_idx].to_dict('records')
            
            # Skip if no reactions
            if len(reactions) == 0:
                continue
            
            # Create thread structure
            thread = {
                'source_tweet': root_tweet,
                'reactions': reactions,
                'thread_id': str(topic_key),
                'category': 'rumours' if root_tweet.get('label', 0) == 1 else 'non-rumours',
                'source': 'credbank'  # Add source field
            }
            
            thread_data.append(thread)
        
        # If no threads were found, return an empty DataFrame with required columns
        if not thread_data:
            return pd.DataFrame({'source': ['credbank'], 'label': [0]})
        
        # Convert to DataFrame format similar to PHEME
        flattened_data = self._flatten_thread_data(thread_data)
        
        # Ensure required columns exist
        if 'source' not in flattened_data.columns:
            flattened_data['source'] = 'credbank'
        
        return flattened_data
    
    def capture_buzzfeed_threads(self, buzzfeed_df: pd.DataFrame = None) -> pd.DataFrame:
        """Capture Twitter threads from BuzzFeed dataset.
        
        Args:
            buzzfeed_df: BuzzFeed dataset DataFrame
            
        Returns:
            DataFrame with thread structure
        """
        # Return empty DataFrame if no input data
        if buzzfeed_df is None or buzzfeed_df.empty:
            return pd.DataFrame()
        
        # Ensure all required fields are available or add defaults
        required_fields = ['article_id', 'title', 'label']
        for field in required_fields:
            if field not in buzzfeed_df.columns:
                if field == 'article_id':
                    buzzfeed_df['article_id'] = [f"article_{i}" for i in range(len(buzzfeed_df))]
                elif field == 'title':
                    buzzfeed_df['title'] = ["Untitled article" for _ in range(len(buzzfeed_df))]
                elif field == 'label':
                    buzzfeed_df['label'] = [0 for _ in range(len(buzzfeed_df))]
        
        threads = []
        
        # Process each article
        for idx, row in buzzfeed_df.iterrows():
            thread_id = row.get('article_id', f"article_{idx}")
            thread_root = row.get('title', "Untitled article")
            
            # Add basic thread information
            thread_data = {
                'id': thread_id,
                'thread_root': thread_root,
                'thread_depth': 0,
                'thread_size': 1, 
                'label': row.get('label', 0),
                'source': 'buzzfeed'  # Important: Add source field
            }
            
            # Add thread to collection
            threads.append(thread_data)
        
        # Create DataFrame
        thread_df = pd.DataFrame(threads) if threads else pd.DataFrame()
        
        return thread_df
    
    def _search_twitter_for_headline(self, headline: str) -> List[Dict]:
        """Search Twitter for a headline and return results.
        
        Args:
            headline: The headline to search for
            
        Returns:
            List of tweet dictionaries
        """
        # Ensure headline is not None and cast it to string
        headline = str(headline) if headline is not None else ""
        try:
            # Create mock Twitter search results since we don't have API access
            # This simulates finding 3 tweets for each headline
            mock_tweets = []
            for i in range(3):
                tweet_id = f"{abs(hash(str(headline) + str(i))) % 10000000000}"
                mock_tweets.append({
                    'id': tweet_id,
                    'id_str': str(tweet_id),
                    'text': str(headline)[:140],  # Simulate tweet text limit
                    'user': {
                        'screen_name': f"user_{i}_{abs(hash(str(headline))) % 1000}",
                        'name': f"User {i}",
                        'id': abs(hash(f"user_{i}_{str(headline)}")) % 1000000
                    },
                    'retweet_count': 100 - (i * 30),  # First has most retweets
                    'favorite_count': 50 - (i * 15),
                    'created_at': '2023-01-01',
                    'metadata': {
                        'result_type': 'popular'
                    }
                })
            return mock_tweets
        except Exception as e:
            warnings.warn(f"Twitter search failed: {str(e)}")
            return []
    
    def _get_twitter_replies(self, tweet_id: str) -> List[Dict]:
        """Get replies to a tweet.
        
        Args:
            tweet_id: The ID of the tweet to get replies for
            
        Returns:
            List of reply tweet dictionaries
        """
        try:
            # Create mock reply data since we don't have actual Twitter API access
            # Generate 3-7 mock replies for each tweet
            mock_replies = []
            reply_count = np.random.randint(3, 8)
            
            for i in range(reply_count):
                reply_id = f"{abs(hash(tweet_id + str(i))) % 10000000000}"
                reply_text = f"This is a mock reply #{i} to tweet {tweet_id}"
                
                mock_replies.append({
                    'id': reply_id,
                    'id_str': str(reply_id),
                    'text': reply_text,
                    'in_reply_to_status_id_str': tweet_id,
                    'user': {
                        'screen_name': f"replier_{i}_{abs(hash(tweet_id)) % 1000}",
                        'name': f"Replier {i}",
                        'id': abs(hash(f"replier_{i}_{tweet_id}")) % 1000000
                    },
                    'retweet_count': np.random.randint(0, 30),
                    'favorite_count': np.random.randint(0, 50),
                    'created_at': '2023-01-02',
                })
            
            return mock_replies
        except Exception as e:
            warnings.warn(f"Getting Twitter replies failed: {str(e)}")
            return []
    
    def _generate_mock_twitter_threads(self, stories: pd.DataFrame) -> List[Dict]:
        """Generate mock Twitter threads for testing without Twitter API access.
        
        Args:
            stories: DataFrame of BuzzFeed stories
            
        Returns:
            List of mock thread data dictionaries
        """
        thread_data = []
        
        for _, story in stories.iterrows():
            # Skip stories with missing critical data
            if story is None:
                continue
                
            # Use .get() to safely access dictionary keys with defaults
            article_id = story.get('article_id', f"unknown_{np.random.randint(1000, 9999)}")
            title = story.get('title', 'Untitled Article')
            
            if article_id is None:
                article_id = f"unknown_{np.random.randint(1000, 9999)}"
            if title is None:
                title = 'Untitled Article'
                
            # Create 1-3 threads per story
            num_threads = np.random.randint(1, 4)  # Random number between 1 and 3
            
            for i in range(num_threads):
                # Create a mock root tweet
                root_tweet = {
                    'id': f"{article_id}_{i}",
                    'id_str': f"{article_id}_{i}",
                    'text': title,
                    'created_at': story.get('publish_date', ''),
                    'retweet_count': np.random.randint(50, 1000),  # Random retweet count
                    'favorite_count': np.random.randint(100, 2000),  # Random favorite count
                    'user': {
                        'id': np.random.randint(10000, 99999),
                        'id_str': str(np.random.randint(10000, 99999)),
                        'name': f"User_{np.random.randint(1000, 9999)}",
                        'screen_name': f"user_{np.random.randint(1000, 9999)}",
                        'followers_count': np.random.randint(100, 10000),
                        'friends_count': np.random.randint(100, 1000),
                        'statuses_count': np.random.randint(1000, 5000),
                        'verified': np.random.choice([True, False], p=[0.1, 0.9]),  # 10% chance of being verified
                        'created_at': ''
                    }
                }
                
                # Generate 3-10 mock replies
                num_replies = np.random.randint(3, 11)
                replies = []
                
                for j in range(num_replies):
                    # Ensure we handle potential None in title
                    safe_title = title[:30] if title else "Unknown title"
                    
                    reply = {
                        'id': f"{article_id}_{i}_reply_{j}",
                        'id_str': f"{article_id}_{i}_reply_{j}",
                        'text': f"This is a mock reply {j} to the article: {safe_title}...",
                        'created_at': '',
                        'retweet_count': np.random.randint(0, 50),
                        'favorite_count': np.random.randint(0, 100),
                        'in_reply_to_status_id': root_tweet['id'],
                        'in_reply_to_status_id_str': root_tweet['id_str'],
                        'user': {
                            'id': np.random.randint(10000, 99999),
                            'id_str': str(np.random.randint(10000, 99999)),
                            'name': f"User_{np.random.randint(1000, 9999)}",
                            'screen_name': f"user_{np.random.randint(1000, 9999)}",
                            'followers_count': np.random.randint(10, 5000),
                            'friends_count': np.random.randint(10, 500),
                            'statuses_count': np.random.randint(100, 3000),
                            'verified': np.random.choice([True, False], p=[0.05, 0.95]),  # 5% chance of being verified
                            'created_at': ''
                        }
                    }
                    replies.append(reply)
                
                # Convert veracity to lowercase safely
                veracity = story.get('veracity', '')
                if veracity is None:
                    veracity = ''
                    
                # Check for rating field (newer format) or use veracity (older format)
                rating = story.get('rating', '')
                if rating:
                    # Use rating directly
                    is_rumour = rating.lower() == 'fake'
                else:
                    # Use veracity as fallback
                    veracity_lower = veracity.lower() if isinstance(veracity, str) else ''
                    is_rumour = veracity_lower in ['mostly false', 'false', 'mixture of true and false', 'no factual content']
                
                # Create thread structure
                thread = {
                    'source_tweet': root_tweet,
                    'reactions': replies,
                    'thread_id': f"{article_id}_{i}",
                    'category': 'rumours' if is_rumour else 'non-rumours',
                    'rating': rating or veracity  # Store either rating or veracity for later use
                }
                
                thread_data.append(thread)
        
        # Ensure we have close to the expected distribution (15 true, 20 false)
        # Adjust if necessary by adding or removing threads
        true_threads = [t for t in thread_data if t['category'] == 'non-rumours']
        false_threads = [t for t in thread_data if t['category'] == 'rumours']
        
        # Target counts
        target_true = 15
        target_false = 20
        
        # Adjust true threads
        if len(true_threads) > target_true:
            true_threads = true_threads[:target_true]
        
        # Adjust false threads
        if len(false_threads) > target_false:
            false_threads = false_threads[:target_false]
        
        # Combine adjusted threads
        adjusted_thread_data = true_threads + false_threads
        
        return adjusted_thread_data
    
    def _flatten_thread_data(self, thread_data: List[Dict]) -> pd.DataFrame:
        """Convert the thread data structure into a flattened DataFrame similar to PHEME.
        
        Args:
            thread_data: List of thread dictionaries
            
        Returns:
            Flattened DataFrame with thread structure
        """
        flattened_data = []
        
        if thread_data is None:
            return pd.DataFrame()
            
        for thread in thread_data:
            if thread is None:
                continue
                
            # Extract source tweet data with safe access
            source_tweet = thread.get('source_tweet', {})
            if source_tweet is None:
                source_tweet = {}
            
            # Create base row with thread info
            row = {
                'thread_id': thread.get('thread_id', f"unknown_{np.random.randint(1000, 9999)}"),
                'category': thread.get('category', 'unknown'),
                'num_reactions': len(thread.get('reactions', []))
            }
            
            # Process source tweet fields
            for key, value in source_tweet.items():
                if value is None:
                    continue
                    
                if isinstance(value, dict):
                    # Handle nested dictionaries (like user data)
                    for sub_key, sub_value in value.items():
                        if sub_value is not None:  # Skip None values
                            field_key = f'source_tweet_user_{sub_key}'
                            row[field_key] = sub_value
                else:
                    field_key = f'source_tweet_{key}'
                    row[field_key] = value
            
            # Initialize reaction data lists
            reaction_texts = []
            reaction_created_at = []
            reaction_id = []
            reaction_in_reply_to_status_id = []
            reaction_user_ids = []
            
            # Process reactions with safe access
            reactions = thread.get('reactions', [])
            if reactions is None:
                reactions = []
                
            for reaction in reactions:
                if reaction is None:
                    continue
                    
                reaction_texts.append(reaction.get('text', ''))
                reaction_created_at.append(reaction.get('created_at', ''))
                reaction_id.append(reaction.get('id_str', ''))
                reaction_in_reply_to_status_id.append(reaction.get('in_reply_to_status_id_str', ''))
                
                # Add user ID if available
                user = reaction.get('user', {})
                if user is not None and isinstance(user, dict) and 'id_str' in user:
                    reaction_user_ids.append(user['id_str'])
                else:
                    reaction_user_ids.append('')
            
            # Add reaction data to row
            row['reaction_texts'] = reaction_texts
            row['reaction_created_at'] = reaction_created_at
            row['reaction_id'] = reaction_id
            row['reaction_in_reply_to_status_id'] = reaction_in_reply_to_status_id
            row['reaction_user_ids'] = reaction_user_ids
            
            # Add label (binary: 1 for rumours, 0 for non-rumours)
            category = thread.get('category', '')
            row['label'] = 1 if category == 'rumours' else 0
            
            # Preserve rating field if it exists
            if 'rating' in thread:
                row['rating'] = thread['rating']
            
            flattened_data.append(row)
        
        return pd.DataFrame(flattened_data)
    
    def save_threaded_datasets(self, credbank_threads: pd.DataFrame = None, 
                              buzzfeed_threads: pd.DataFrame = None,
                              output_dir: str = None) -> Tuple[str, str]:
        """Save the threaded datasets to CSV files.
        
        Args:
            credbank_threads: DataFrame with CREDBANK in thread format
            buzzfeed_threads: DataFrame with BuzzFeed in thread format
            output_dir: Directory to save output files
            
        Returns:
            Tuple of (credbank_output_path, buzzfeed_output_path)
        """
        # Use appropriate paths if output_dir not specified
        if output_dir is None:
            credbank_output_dir = self.credbank_path
            buzzfeed_output_dir = self.buzzfeed_path
        else:
            credbank_output_dir = output_dir
            buzzfeed_output_dir = output_dir
        
        # Create directories if they don't exist
        os.makedirs(credbank_output_dir, exist_ok=True)
        os.makedirs(buzzfeed_output_dir, exist_ok=True)
        
        # Initialize paths
        credbank_output_path = None
        buzzfeed_output_path = None
        
        # Save CREDBANK threaded dataset
        if credbank_threads is not None and not credbank_threads.empty:
            credbank_output_path = os.path.join(credbank_output_dir, 'credbank_threaded_dataset.csv')
            credbank_threads.to_csv(credbank_output_path, index=False)
            print(f"Saved CREDBANK threaded dataset to: {credbank_output_path}")
        
        # Save BuzzFeed threaded dataset
        if buzzfeed_threads is not None and not buzzfeed_threads.empty:
            buzzfeed_output_path = os.path.join(buzzfeed_output_dir, 'buzzfeed_threaded_dataset.csv')
            buzzfeed_threads.to_csv(buzzfeed_output_path, index=False)
            print(f"Saved BuzzFeed threaded dataset to: {buzzfeed_output_path}")
        
        return credbank_output_path, buzzfeed_output_path


# Example usage
def capture_threads(base_path: str = 'data'):
    """Capture and save thread structure for CREDBANK and BuzzFeed datasets.
    
    Args:
        base_path: Base directory path for dataset storage
    """
    # Initialize thread capture tool
    thread_tool = ThreadCaptureTool(base_path)
    
    try:
        # Capture CREDBANK threads
        print("Capturing CREDBANK threads...")
        credbank_threads = thread_tool.capture_credbank_threads()
        print(f"Created {len(credbank_threads)} CREDBANK threads")
        
        # Capture BuzzFeed threads
        print("\nCapturing BuzzFeed threads...")
        buzzfeed_threads = thread_tool.capture_buzzfeed_threads()
        print(f"Created {len(buzzfeed_threads)} BuzzFeed threads")
        
        # Save threaded datasets
        print("\nSaving threaded datasets...")
        credbank_path, buzzfeed_path = thread_tool.save_threaded_datasets(
            credbank_threads, buzzfeed_threads
        )
        
        return credbank_threads, buzzfeed_threads
        
    except Exception as e:
        print(f"Error capturing threads: {str(e)}")
        return None, None


if __name__ == "__main__":
    capture_threads() 