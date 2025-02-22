import json
import os
import pandas as pd
import sweetviz as sv
from typing import Dict, List, Tuple, Any, Union
from collections import defaultdict
import warnings
from utils.features import (
    PhemeStructuralFeatureExtractor,
    PhemeUserFeatureExtractor,
    PhemeContentFeatureExtractor,
    PhemeTemporalFeatureExtractor
)
from utils.dataset_alignment import save_feature_sets

# Loads a single tweet's data from its JSON file
def load_tweet(file_path: str) -> Dict:
    """Load a single tweet from a JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Loads a source tweet and its reaction tweets from a conversation thread directory
def load_thread(thread_path: str) -> Tuple[Dict, List[Dict]]:
    """Load a source tweet and its associated reaction tweets from a thread directory"""
    # Get the source tweet from source-tweet directory
    source_tweet_path = os.path.join(thread_path, 'source-tweet')
    source_tweet_file = os.listdir(source_tweet_path)[0]
    source_tweet = load_tweet(os.path.join(source_tweet_path, source_tweet_file))
    
    # Get all reaction tweets if they exist
    reactions = []
    reactions_path = os.path.join(thread_path, 'reactions')
    if os.path.exists(reactions_path):
        for reaction_file in os.listdir(reactions_path):
            reaction = load_tweet(os.path.join(reactions_path, reaction_file))
            reactions.append(reaction)
    
    return source_tweet, reactions

# Converts nested PHEME dataset structure into a flat DataFrame format
def create_flattened_dataframe(pheme_dataset_raw: Dict) -> pd.DataFrame:
    """Convert the raw PHEME dataset into a flattened DataFrame structure"""
    flattened_data = []
    
    for event in pheme_dataset_raw:
        for category in ['rumours', 'non-rumours']:
            if category not in pheme_dataset_raw[event]:
                continue
                
            for thread in pheme_dataset_raw[event][category]:
                # Extract source tweet data
                source_tweet = thread['source_tweet']
                
                # Create base row with thread info
                row = {
                    'event': event,
                    'category': category,
                    'thread_id': str(thread['thread_id']),
                    'num_reactions': len(thread['reactions'])
                }
                
                # Process source tweet fields
                for key, value in source_tweet.items():
                    if isinstance(value, dict):
                        # Handle nested dictionaries
                        for sub_key, sub_value in value.items():
                            field_key = f'source_tweet_{key}_{sub_key}'
                            row[field_key] = process_field_value(sub_value)
                    else:
                        field_key = f'source_tweet_{key}'
                        row[field_key] = process_field_value(value)
                
                # Process reaction data
                reaction_data = process_reactions(thread['reactions'])
                row.update(reaction_data)
                flattened_data.append(row)
    
    return pd.DataFrame(flattened_data)

# Processes field values while preserving their data types and handling edge cases
def process_field_value(value: Any) -> Any:
    """Process a field value maintaining its appropriate type"""
    if value is None:
        return None
    elif value == "":
        return ""
    elif isinstance(value, (bool, int, float)):
        return value
    return str(value)

# Extracts relevant metrics from reaction tweets into a structured dictionary
def process_reactions(reactions: List[Dict]) -> Dict:
    """Extract and process reaction data from a list of reactions"""
    return {
        'reaction_texts': [reaction.get('text', '') for reaction in reactions],
        'reaction_favorite_counts': [int(reaction.get('favorite_count', 0)) for reaction in reactions],
        'reaction_retweet_counts': [int(reaction.get('retweet_count', 0)) for reaction in reactions],
        'reaction_created_at': [reaction.get('created_at', '') for reaction in reactions],
        'reaction_user_ids': [reaction.get('user', {}).get('id_str', '') for reaction in reactions]
    }

# PHEME dataset processor for rumor detection using social media conversation threads
def load_pheme_dataset_raw(base_path: str = 'data/pheme/pheme-rnr-dataset', output_dir: str = None, save_csv: bool = False) -> pd.DataFrame:
    """Load and process the raw PHEME dataset from the given path.
    Returns the raw dataset structure as a flattened DataFrame.
    
    Args:
        base_path: Path to the PHEME dataset directory
        output_dir: Directory to save output files (defaults to base_path if None)
        save_csv: Whether to save intermediate CSV files
        
    Returns:
        DataFrame containing the raw dataset structure
    """
    # Use base_path as output_dir if not specified
    output_dir = output_dir or base_path
    
    # List of event categories in PHEME dataset
    events = ['charliehebdo', 'ferguson', 'germanwings-crash', 'ottawashooting', 'sydneysiege']
    dataset = defaultdict(lambda: {'rumours': [], 'non-rumours': []})
    
    # Verify dataset path exists
    abs_base_path = os.path.abspath(base_path)
    print(f"Loading dataset from: {abs_base_path}")
    
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Dataset directory not found at: {abs_base_path}")
    
    # Process each event category
    for event in events:
        event_path = os.path.join(base_path, event)
        if not os.path.exists(event_path):
            print(f"Warning: Event directory not found: {event_path}")
            continue
            
        # Process rumours and non-rumours separately
        for category in ['rumours', 'non-rumours']:
            category_path = os.path.join(event_path, category)
            if not os.path.exists(category_path):
                print(f"Warning: Category directory not found: {category_path}")
                continue
                
            print(f"Processing {event}/{category}...")
            for thread_id in os.listdir(category_path):
                # Skip hidden files
                if thread_id.startswith('.'):
                    continue
                    
                thread_path = os.path.join(category_path, thread_id)
                try:
                    source_tweet, reactions = load_thread(thread_path)
                    dataset[event][category].append({
                        'source_tweet': source_tweet,
                        'reactions': reactions,
                        'thread_id': thread_id
                    })
                except Exception as e:
                    print(f"Error processing thread {thread_id}: {str(e)}")
    
    # Convert to DataFrame
    df = create_flattened_dataframe(dataset)
    
    # Save if requested
    if save_csv:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'pheme_raw_dataset.csv')
        df.to_csv(output_path, index=False)
        print(f"Saved raw dataset to: {output_path}")
    
    return df

# Processes raw PHEME data into a structured DataFrame with basic conversation features
def load_pheme_dataset_extended(raw_dataset: Union[pd.DataFrame, str, None] = None, base_path: str = 'data/pheme/pheme-rnr-dataset', output_dir: str = None, save_csv: bool = False) -> pd.DataFrame:
    """Load and process the PHEME dataset with basic features but without the feature extractors.
    Can either take a raw dataset DataFrame or load from a CSV file.
    
    Args:
        raw_dataset: Either a DataFrame from load_pheme_dataset_raw or a path to a CSV file
        base_path: Path to the PHEME dataset directory (used if raw_dataset is None)
        output_dir: Directory to save output files (defaults to base_path if None)
        save_csv: Whether to save intermediate CSV files
        
    Returns:
        DataFrame containing the processed dataset with basic features
    """
    # Use base_path as output_dir if not specified
    output_dir = output_dir or base_path
    
    if isinstance(raw_dataset, str) and os.path.exists(raw_dataset):
        # Load from specified CSV path
        print(f"Loading extended dataset from CSV: {raw_dataset}")
        return pd.read_csv(raw_dataset)
    
    if raw_dataset is None:
        raw_dataset = load_pheme_dataset_raw(base_path)
    
    # Process the raw DataFrame
    df = raw_dataset.copy()
    
    # Add binary label (1 for rumours, 0 for non-rumours)
    df['label'] = (df['category'] == 'rumours').astype(int)
    
    # Save if requested
    if save_csv:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'pheme_extended_dataset.csv')
        df.to_csv(output_path, index=False)
        print(f"Saved extended dataset to: {output_path}")
    
    return df

# Extracts advanced features from PHEME conversations using specialized feature extractors
def load_pheme_features_dataset(extended_dataset: Union[pd.DataFrame, str, None] = None, base_path: str = 'data/pheme/pheme-rnr-dataset', output_dir: str = None, save_csv: bool = False, include_additional_features: bool = False) -> pd.DataFrame:
    """Load or create the PHEME features dataset with all feature extractors applied.
    Can either take an extended dataset DataFrame or load from a CSV file.
    
    Args:
        extended_dataset: Either a DataFrame from load_pheme_dataset_extended or a path to a CSV file
        base_path: Path to the PHEME dataset directory (used if extended_dataset is None)
        output_dir: Directory to save output files (defaults to base_path if None)
        save_csv: Whether to save intermediate CSV files
        include_additional_features: Whether to include additional features
        
    Returns:
        DataFrame containing only the extracted features, source, and label
    """
    # Use base_path as output_dir if not specified
    output_dir = output_dir or base_path
    
    if isinstance(extended_dataset, str) and os.path.exists(extended_dataset):
        # Load from specified CSV path
        print(f"Loading features dataset from CSV: {extended_dataset}")
        return pd.read_csv(extended_dataset)
    
    if extended_dataset is None:
        extended_dataset = load_pheme_dataset_extended(base_path=base_path, output_dir=output_dir, save_csv=save_csv)
    
    # Extract all features
    df_with_features = extract_all_features(extended_dataset, include_additional_features)
    
    # Get feature columns
    feature_columns = [col for col in df_with_features.columns if any(
        col.startswith(prefix) for prefix in 
        ['structural_', 'user_', 'content_', 'temporal_']
    )]
    
    # Create features DataFrame
    features_df = df_with_features[feature_columns].copy()
    features_df.insert(0, 'source', 'pheme')
    features_df['label'] = df_with_features['label']
    
    # Save if requested
    if save_csv:
        os.makedirs(output_dir, exist_ok=True)
        save_feature_sets(features_df, output_dir, 'pheme')
    
    return features_df

# Applies all PHEME-specific feature extractors to generate the complete feature set
def extract_all_features(df: pd.DataFrame, include_additional_features: bool = False) -> pd.DataFrame:
    """Extract all PHEME-specific features from the dataset.
    
    Args:
        df: Input DataFrame
        include_additional_features: Whether to include additional features not in the paper
        
    Returns:
        DataFrame with all extracted features
    """
    # Make a copy of the input DataFrame to avoid modifying the original
    result_df = df.copy()
    
    # Initialize feature extractors
    extractors = [
        PhemeStructuralFeatureExtractor(result_df),
        PhemeUserFeatureExtractor(result_df, include_additional_features),
        PhemeContentFeatureExtractor(result_df, include_additional_features),
        PhemeTemporalFeatureExtractor(result_df, include_additional_features)
    ]
    
    # Apply each extractor and update the DataFrame
    for extractor in extractors:
        # Get features from the current extractor
        updated_df = extractor.extract_features()
        
        # Add any new columns from the updated DataFrame
        new_columns = set(updated_df.columns) - set(result_df.columns)
        for col in new_columns:
            result_df[col] = updated_df[col]
    
    return result_df

# Prepares DataFrame for Sweetviz analysis by converting complex data types to simple ones
def prepare_df_for_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare DataFrame for Sweetviz analysis by handling complex data types"""
    analysis_df = df.copy()
    for column in analysis_df.columns:
        # Convert list columns to their lengths
        if analysis_df[column].apply(lambda x: isinstance(x, list)).any():
            analysis_df[column] = analysis_df[column].apply(lambda x: len(x) if isinstance(x, list) else x)
        # Convert unhashable types to strings
        elif analysis_df[column].apply(lambda x: isinstance(x, (dict, set))).any():
            analysis_df[column] = analysis_df[column].apply(str)
    return analysis_df

# Main execution function to process PHEME dataset and generate analysis reports
def main():
    """Main execution function"""
    warnings.filterwarnings('ignore')

    # Set output directory
    output_dir = 'data/pheme'
    os.makedirs(output_dir, exist_ok=True)

    # Load raw dataset
    print("Loading raw dataset...")
    raw_df = load_pheme_dataset_raw(output_dir=output_dir, save_csv=True)
    
    # Load extended dataset
    print("\nCreating extended dataset...")
    extended_df = load_pheme_dataset_extended(raw_df, output_dir=output_dir, save_csv=True)
    
    # Extract all features once
    print("\nExtracting features...")
    all_features_df = load_pheme_features_dataset(extended_df, output_dir=output_dir, save_csv=False, include_additional_features=True)

    # Split into paper and all features using dataset_alignment.save_feature_sets
    paper_features_df, all_features_df = save_feature_sets(all_features_df, output_dir, 'pheme')

    # Generate analysis report
    try:
        analysis_df = prepare_df_for_analysis(paper_features_df)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            sweet_report = sv.analyze(analysis_df)
            output_path = os.path.join(output_dir, 'pheme_paper_features_analysis_report.html')
            sweet_report.show_html(output_path) # Could use sweet_report.show_notebook()
            print(f"Analysis report saved to: {output_path}")
    except Exception as e:
        print(f"Warning: Could not generate Sweetviz report. Error: {str(e)}")

    # Generate analysis report
    try:
        analysis_df = prepare_df_for_analysis(all_features_df)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            sweet_report = sv.analyze(analysis_df)
            output_path = os.path.join(output_dir, 'pheme_all_features_analysis_report.html')
            sweet_report.show_html(output_path) # Could use sweet_report.show_notebook()
            print(f"Analysis report saved to: {output_path}")
    except Exception as e:
        print(f"Warning: Could not generate Sweetviz report. Error: {str(e)}")
    
    print("\nDataset processing complete!")
    print(f"Raw dataset shape: {raw_df.shape}")
    print(f"Extended dataset shape: {extended_df.shape}")
    print(f"Paper features shape: {paper_features_df.shape}")
    print(f"All features shape: {all_features_df.shape}")

if __name__ == "__main__":
    main()