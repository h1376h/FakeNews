import requests
import json
import pandas as pd
import os
import numpy as np
from joblib import load

def check_file_exists(file_path):
    """Check if a file exists and print its info."""
    if os.path.exists(file_path):
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
        print(f"File {file_path} exists ({file_size:.2f} MB)")
        
        # If it's a CSV, show the number of rows and columns
        if file_path.endswith('.csv'):
            try:
                df = pd.read_csv(file_path)
                print(f" - Contains {len(df)} rows and {len(df.columns)} columns")
                print(f" - Column names: {', '.join(df.columns[:5])}...")
            except Exception as e:
                print(f" - Error reading CSV: {str(e)}")
    else:
        print(f"File {file_path} does not exist.")

def get_model_features(model_path):
    """Get the list of features used by the best model based on the complete_results.txt file."""
    result_dir = os.path.dirname(model_path)
    feature_file = os.path.join(result_dir, "complete_results.txt")
    
    if not os.path.exists(feature_file):
        print(f"Warning: Feature list file {feature_file} not found")
        return None
    
    features = []
    capture = False
    
    with open(feature_file, 'r') as f:
        for line in f:
            if "- Best feature set:" in line:
                capture = True
                continue
            elif capture and line.strip() == "":
                break
            elif capture:
                features.append(line.strip())
    
    if features:
        print(f"Loaded {len(features)} model features from {feature_file}")
        return features
    
    return None

# Example 1: Using the local Flask API
def test_local_api(csv_path, row_index=0):
    """Test the local Flask API."""
    print("\n=== Testing Local API ===")
    
    # Check if the CSV file exists
    check_file_exists(csv_path)
    
    # API endpoint (local server)
    url = "http://127.0.0.1:5000/classify"
    
    # Prepare the request data
    data = {
        "csv_path": csv_path,
        "row_index": row_index
    }
    
    print(f"Sending request to {url} with data: {json.dumps(data, indent=2)}")
    
    try:
        # Send the POST request
        response = requests.post(url, json=data)
        
        # Check if request was successful
        if response.status_code == 200:
            result = response.json()
            print("\nClassification Result:")
            print(f" - Classification: {result['classification']}")
            print(f" - Probability Fake: {result['probability_fake']:.4f}")
            print(f" - Probability Real: {result['probability_real']:.4f}")
            print(f" - Confidence: {result['confidence']:.4f}")
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except requests.exceptions.ConnectionError:
        print("Connection Error: Make sure the Flask API is running (python app-local.py)")
    except Exception as e:
        print(f"Error: {str(e)}")

# Example 2: Direct Python example (simulating the API response)
def simulate_api_response(csv_path, row_index=0):
    """Simulate the API response using Python directly."""
    print("\n=== Simulating API Response ===")
    
    try:
        # Check if required files exist
        check_file_exists(csv_path)
        
        # Load the model (similar to the API code)
        model_paths = [
            'results/pheme_paper_elimination/best_classifier.joblib',
            'results/elimination/best_classifier.joblib'
        ]
        
        model = None
        model_path = None
        for path in model_paths:
            check_file_exists(path)
            if os.path.exists(path):
                try:
                    model = load(path)
                    model_path = path
                    print(f"Model loaded from {path}")
                    break
                except Exception as e:
                    print(f"Error loading model from {path}: {str(e)}")
        
        if model is None:
            print("Error: Could not load model")
            return
        
        # Get the features used by the model
        model_features = get_model_features(model_path)
        if model_features is None:
            print("Error: Could not determine which features the model uses")
            return
        
        # Load the dataset
        if not os.path.exists(csv_path):
            print(f"Error: CSV file {csv_path} not found")
            return
        
        df = pd.read_csv(csv_path)
        
        # Check if row index is valid
        if row_index < 0 or row_index >= len(df):
            print(f"Error: Invalid row index {row_index} for CSV with {len(df)} rows")
            return
        
        # Extract row data
        row_data = df.iloc[row_index]
        
        # Check if all required features exist
        missing_features = [feat for feat in model_features if feat not in df.columns]
        if missing_features:
            print(f"Error: Missing required features in CSV: {', '.join(missing_features)}")
            return
        
        # Prepare features using only the ones the model was trained on
        features = row_data[model_features].fillna(df[model_features].median())
        
        # Convert to numpy array for prediction
        X = np.array([features.to_numpy()])
        
        # Make prediction
        prediction = model.predict_proba(X)[0]
        
        # Prepare response (similar to API response)
        result = {
            'row_index': row_index,
            'probability_fake': float(prediction[1]),
            'probability_real': float(prediction[0]),
            'classification': 'FAKE' if prediction[1] > 0.5 else 'REAL',
            'confidence': float(max(prediction))
        }
        
        print("\nClassification Result:")
        print(f" - Classification: {result['classification']}")
        print(f" - Probability Fake: {result['probability_fake']:.4f}")
        print(f" - Probability Real: {result['probability_real']:.4f}")
        print(f" - Confidence: {result['confidence']:.4f}")
    
    except Exception as e:
        print(f"Error: {str(e)}")

# Example 3: Using the Google Colab API (requires ngrok URL)
def test_colab_api(ngrok_url, csv_path, row_index=0):
    """Test the Google Colab API with ngrok URL."""
    print("\n=== Testing Google Colab API ===")
    
    # Check if the CSV file exists
    check_file_exists(csv_path)
    
    # API endpoint (ngrok URL)
    url = f"{ngrok_url}/classify"
    
    # Prepare the request data
    data = {
        "csv_path": csv_path,
        "row_index": row_index
    }
    
    print(f"Sending request to {url} with data: {json.dumps(data, indent=2)}")
    
    try:
        # Send the POST request
        response = requests.post(url, json=data)
        
        # Check if request was successful
        if response.status_code == 200:
            result = response.json()
            print("\nClassification Result:")
            print(f" - Classification: {result['classification']}")
            print(f" - Probability Fake: {result['probability_fake']:.4f}")
            print(f" - Probability Real: {result['probability_real']:.4f}")
            print(f" - Confidence: {result['confidence']:.4f}")
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except requests.exceptions.ConnectionError:
        print("Connection Error: Make sure the ngrok URL is correct and the Flask API is running")
    except Exception as e:
        print(f"Error: {str(e)}")

# Main section to run examples
if __name__ == "__main__":
    # Configuration for the examples
    csv_path = "data/pheme/pheme_paper_features.csv"
    row_index = 42
    
    # Example 1: Test Local API (uncomment to use)
    # Note: Make sure the Flask API is running with 'python app-local.py'
    # test_local_api(csv_path, row_index)
    
    # Example 2: Simulate API response
    simulate_api_response(csv_path, row_index)
    
    # Example 3: Test Google Colab API (uncomment and update URL to use)
    # Note: Replace 'YOUR_NGROK_URL' with the actual URL from Google Colab
    # test_colab_api("YOUR_NGROK_URL", csv_path, row_index)