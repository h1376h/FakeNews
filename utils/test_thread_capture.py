import os
import sys
import pandas as pd
import numpy as np

# Import ThreadCaptureTool
from thread_capture import ThreadCaptureTool

def test_mock_thread_generation():
    """Test the mock thread generation with various inputs including None values."""
    
    # Create the thread capture tool
    tool = ThreadCaptureTool()
    
    # Test Case 1: Normal data
    print("Test Case 1: Normal data")
    normal_df = pd.DataFrame([
        {'article_id': '123', 'title': 'Test Article', 'veracity': 'mostly false'}
    ])
    
    try:
        threads = tool._generate_mock_twitter_threads(normal_df)
        print(f"Success! Generated {len(threads)} threads")
        print(f"First thread category: {threads[0]['category'] if threads else 'N/A'}\n")
    except Exception as e:
        print(f"Error: {str(e)}\n")
    
    # Test Case 2: Data with None values
    print("Test Case 2: Data with None values")
    none_df = pd.DataFrame([
        {'article_id': None, 'title': None, 'veracity': None}
    ])
    
    try:
        threads = tool._generate_mock_twitter_threads(none_df)
        print(f"Success! Generated {len(threads)} threads")
        print(f"First thread category: {threads[0]['category'] if threads else 'N/A'}\n")
    except Exception as e:
        print(f"Error: {str(e)}\n")
    
    # Test Case 3: Mixed data
    print("Test Case 3: Mixed data")
    mixed_df = pd.DataFrame([
        {'article_id': '123', 'title': 'Test Article', 'veracity': 'mostly false'},
        {'article_id': None, 'title': None, 'veracity': None},
        {'article_id': '456', 'title': 'Another Test', 'veracity': 'mostly true'}
    ])
    
    try:
        threads = tool._generate_mock_twitter_threads(mixed_df)
        print(f"Success! Generated {len(threads)} threads")
        print(f"Thread distribution: {len([t for t in threads if t['category'] == 'rumours'])} rumours, "
              f"{len([t for t in threads if t['category'] == 'non-rumours'])} non-rumours\n")
    except Exception as e:
        print(f"Error: {str(e)}\n")
    
    # Test Case 4: Test the flattening function
    print("Test Case 4: Testing _flatten_thread_data")
    try:
        if 'threads' in locals() and threads:
            flattened = tool._flatten_thread_data(threads)
            print(f"Success! Flattened to dataframe with {len(flattened)} rows and {len(flattened.columns)} columns")
            print(f"Column names: {', '.join(flattened.columns[:5])}... (truncated)")
        else:
            print("No threads available to test flattening")
    except Exception as e:
        print(f"Error in flatten_thread_data: {str(e)}")

if __name__ == "__main__":
    test_mock_thread_generation() 