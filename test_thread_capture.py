import os
import pandas as pd
import warnings
from utils.thread_capture import ThreadCaptureTool
from dataset_credbank import load_credbank_dataset_extended, load_credbank_threaded_dataset
from dataset_buzzfeed import load_buzzfeed_dataset_extended, load_buzzfeed_threaded_dataset

def test_thread_capture():
    """Test thread capture functionality for CREDBANK and BuzzFeed datasets."""
    warnings.filterwarnings('ignore')
    
    print("===== Testing Thread Capture Functionality =====\n")
    
    # Set base path
    base_path = 'data'
    
    # Initialize thread capture tool
    print("Initializing thread capture tool...")
    thread_tool = ThreadCaptureTool(base_path)
    
    # Test CREDBANK thread capture
    print("\n----- Testing CREDBANK Thread Capture -----")
    try:
        # Load CREDBANK extended dataset
        print("Loading CREDBANK extended dataset...")
        credbank_df = load_credbank_dataset_extended(base_path=os.path.join(base_path, 'credbank'))
        print(f"Loaded {len(credbank_df)} CREDBANK entries")
        
        # Capture CREDBANK threads
        print("\nCapturing CREDBANK threads...")
        credbank_threads = thread_tool.capture_credbank_threads(credbank_df)
        print(f"Created {len(credbank_threads)} CREDBANK threads")
        print(f"  - Positive samples: {credbank_threads[credbank_threads['label'] == 1].shape[0]}")
        print(f"  - Negative samples: {credbank_threads[credbank_threads['label'] == 0].shape[0]}")
        
        # Test using the main function
        print("\nTesting with main load_credbank_threaded_dataset function...")
        credbank_threads_main = load_credbank_threaded_dataset(credbank_df, 
                                                           base_path=os.path.join(base_path, 'credbank'), 
                                                           save_csv=True)
        print(f"Created {len(credbank_threads_main)} CREDBANK threads with main function")
        
    except Exception as e:
        print(f"Error in CREDBANK thread capture: {str(e)}")
    
    # Test BuzzFeed thread capture
    print("\n----- Testing BuzzFeed Thread Capture -----")
    try:
        # Load BuzzFeed extended dataset
        print("Loading BuzzFeed extended dataset...")
        buzzfeed_df = load_buzzfeed_dataset_extended(base_path=os.path.join(base_path, 'buzzfeed'))
        print(f"Loaded {len(buzzfeed_df)} BuzzFeed entries")
        
        # Capture BuzzFeed threads
        print("\nCapturing BuzzFeed threads...")
        buzzfeed_threads = thread_tool.capture_buzzfeed_threads(buzzfeed_df)
        print(f"Created {len(buzzfeed_threads)} BuzzFeed threads")
        print(f"  - Positive samples (fake): {buzzfeed_threads[buzzfeed_threads['label'] == 1].shape[0]}")
        print(f"  - Negative samples (real): {buzzfeed_threads[buzzfeed_threads['label'] == 0].shape[0]}")
        
        # Test using the main function
        print("\nTesting with main load_buzzfeed_threaded_dataset function...")
        buzzfeed_threads_main = load_buzzfeed_threaded_dataset(buzzfeed_df, 
                                                           base_path=os.path.join(base_path, 'buzzfeed'), 
                                                           save_csv=True)
        print(f"Created {len(buzzfeed_threads_main)} BuzzFeed threads with main function")
        
    except Exception as e:
        print(f"Error in BuzzFeed thread capture: {str(e)}")
    
    print("\n===== Thread Capture Testing Complete =====")

if __name__ == "__main__":
    test_thread_capture() 