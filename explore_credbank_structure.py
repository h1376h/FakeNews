import pandas as pd
import os

def explore_file_structure(file_path: str, n_rows: int = 5, output_file=None):
    """
    Explore the structure of a tab-separated data file.
    
    Args:
        file_path: Path to the data file
        n_rows: Number of rows to read for exploration
        output_file: File object to write output to (optional)
    """
    def print_both(*args, **kwargs):
        print(*args, **kwargs)
        if output_file:
            print(*args, **kwargs, file=output_file)

    print_both(f"\nExploring file: {os.path.basename(file_path)}")
    print_both("=" * 80)
    
    try:
        # Read just a few rows
        data = pd.read_csv(file_path, sep='\t', nrows=n_rows)
        
        print_both("\nColumns (tab-separated):")
        print_both('\t'.join(data.columns))
        
        print_both("\nFirst row (tab-separated):")
        print_both('\t'.join(map(str, data.iloc[0])))
        
        print_both("\nColumns:")
        for col in data.columns:
            print_both(f"- {col}")
        
        print_both("\nData types:")
        print_both(data.dtypes)
        
        print_both("\nFile size:", os.path.getsize(file_path) / (1024 * 1024), "MB")
        
    except Exception as e:
        print_both(f"Error reading file: {str(e)}")

def main():
    # Base directory for CREDBANK data
    base_dir = 'data/credbank/CREDBANK'
    
    # Create output file
    with open('credbank_structure_analysis.txt', 'w') as output_file:
        # List of files to explore
        files = [
            'eventNonEvent_annotations.data',
            'cred_event_TurkRatings.data',
            'cred_event_SearchTweets.data',
            'stream_tweets_byTimestamp.data'
        ]
        
        # Explore each file
        for file in files:
            file_path = os.path.join(base_dir, file)
            if os.path.exists(file_path):
                explore_file_structure(file_path, output_file=output_file)
            else:
                print_both(f"\nFile not found: {file}", file=output_file)

if __name__ == "__main__":
    main() 