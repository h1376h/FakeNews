import json
import os
import pandas as pd
from typing import Dict, List, Tuple, Any, Union
from collections import defaultdict
import numpy as np
import warnings

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
    """
    
    def __init__(self, base_path: str = 'data'):
        """Initialize ThreadCaptureTool.
        
        Args:
            base_path: Base directory path for dataset storage
        """
        self.base_path = base_path
        self.credbank_path = os.path.join(base_path, 'credbank')
        self.buzzfeed_path = os.path.join(base_path, 'buzzfeed')
        self.pheme_path = os.path.join(base_path, 'pheme')
    
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
        grouped_df = credbank_df.groupby('topic_id')
        
        # Initialize list to store thread data
        thread_data = []
        
        # Process each event group
        for topic_id, group in grouped_df:
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
                'thread_id': str(topic_id),
                'category': 'rumours' if root_tweet.get('label', 0) == 1 else 'non-rumours'
            }
            
            thread_data.append(thread)
        
        # Convert to DataFrame format similar to PHEME
        flattened_data = self._flatten_thread_data(thread_data)
        
        return flattened_data
    
    def capture_buzzfeed_threads(self, buzzfeed_df: pd.DataFrame = None) -> pd.DataFrame:
        """Capture threaded structure for BuzzFeed dataset.
        
        Args:
            buzzfeed_df: DataFrame containing BuzzFeed dataset. If None, loads from CSV.
            
        Returns:
            DataFrame with BuzzFeed data in thread structure format
        """
        if buzzfeed_df is None:
            # Load BuzzFeed dataset
            buzzfeed_file = os.path.join(self.buzzfeed_path, 'buzzfeed_extended_dataset.csv')
            if not os.path.exists(buzzfeed_file):
                raise FileNotFoundError(f"BuzzFeed dataset not found at {buzzfeed_file}")
            buzzfeed_df = pd.read_csv(buzzfeed_file)
        
        # Initialize list to store thread data
        thread_data = []
        
        # For each article/headline in BuzzFeed
        for _, row in buzzfeed_df.iterrows():
            # Use headline as root tweet
            root_tweet = {
                'id': row['article_id'],
                'id_str': str(row['article_id']),
                'text': row['title'],
                'created_at': row.get('publish_date', ''),
                'user': {
                    'id': 0,
                    'id_str': '0',
                    'name': row.get('author', 'Unknown'),
                    'screen_name': row.get('author', 'Unknown'),
                    'followers_count': 0,
                    'friends_count': 0,
                    'statuses_count': 0,
                    'verified': False,
                    'created_at': ''
                }
            }
            
            # Use reactions if available
            reactions = []
            reaction_texts = row.get('reaction_texts', [])
            reaction_authors = row.get('reaction_authors', [])
            reaction_timestamps = row.get('reaction_timestamps', [])
            
            # Create reaction tweets
            for i in range(len(reaction_texts)):
                author = reaction_authors[i] if i < len(reaction_authors) else 'Unknown'
                timestamp = reaction_timestamps[i] if i < len(reaction_timestamps) else ''
                
                reaction = {
                    'id': f"{row['article_id']}_r{i}",
                    'id_str': f"{row['article_id']}_r{i}",
                    'text': reaction_texts[i],
                    'created_at': timestamp,
                    'in_reply_to_status_id': row['article_id'],
                    'in_reply_to_status_id_str': str(row['article_id']),
                    'user': {
                        'id': i+1,
                        'id_str': str(i+1),
                        'name': author,
                        'screen_name': author,
                        'followers_count': 0,
                        'friends_count': 0,
                        'statuses_count': 0,
                        'verified': False,
                        'created_at': ''
                    }
                }
                reactions.append(reaction)
            
            # Skip if no reactions
            if len(reactions) == 0:
                continue
            
            # Determine category based on rating
            category = 'rumours' if row.get('rating', '') == 'fake' else 'non-rumours'
            
            # Create thread structure
            thread = {
                'source_tweet': root_tweet,
                'reactions': reactions,
                'thread_id': str(row['article_id']),
                'category': category
            }
            
            thread_data.append(thread)
        
        # Convert to DataFrame format similar to PHEME
        flattened_data = self._flatten_thread_data(thread_data)
        
        return flattened_data
    
    def _flatten_thread_data(self, thread_data: List[Dict]) -> pd.DataFrame:
        """Convert the thread data structure into a flattened DataFrame similar to PHEME.
        
        Args:
            thread_data: List of thread dictionaries
            
        Returns:
            Flattened DataFrame with thread structure
        """
        flattened_data = []
        
        for thread in thread_data:
            # Extract source tweet data
            source_tweet = thread['source_tweet']
            
            # Create base row with thread info
            row = {
                'thread_id': thread['thread_id'],
                'category': thread['category'],
                'num_reactions': len(thread['reactions'])
            }
            
            # Process source tweet fields
            for key, value in source_tweet.items():
                if isinstance(value, dict):
                    # Handle nested dictionaries (like user data)
                    for sub_key, sub_value in value.items():
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
            
            # Process reactions
            for reaction in thread['reactions']:
                reaction_texts.append(reaction.get('text', ''))
                reaction_created_at.append(reaction.get('created_at', ''))
                reaction_id.append(reaction.get('id_str', ''))
                reaction_in_reply_to_status_id.append(reaction.get('in_reply_to_status_id_str', ''))
                
                # Add user ID if available
                if 'user' in reaction and 'id_str' in reaction['user']:
                    reaction_user_ids.append(reaction['user']['id_str'])
                else:
                    reaction_user_ids.append('')
            
            # Add reaction data to row
            row['reaction_texts'] = reaction_texts
            row['reaction_created_at'] = reaction_created_at
            row['reaction_id'] = reaction_id
            row['reaction_in_reply_to_status_id'] = reaction_in_reply_to_status_id
            row['reaction_user_ids'] = reaction_user_ids
            
            # Add label (binary: 1 for rumours, 0 for non-rumours)
            row['label'] = 1 if thread['category'] == 'rumours' else 0
            
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