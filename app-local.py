from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from joblib import load
import os

app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'results/pheme_paper_elimination/best_classifier.joblib'

try:
    if not os.path.exists(MODEL_PATH):
        # Try alternative path as fallback
        alt_path = 'results/elimination/best_classifier.joblib'
        if os.path.exists(alt_path):
            MODEL_PATH = alt_path
            print(f"Using alternative model path: {MODEL_PATH}")
        else:
            raise FileNotFoundError(f"Model not found at {MODEL_PATH} or {alt_path}")
    
    model = load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    # We'll handle this during API requests

# Load feature names that the model was trained on
def get_model_features():
    result_dir = os.path.dirname(MODEL_PATH)
    feature_file = os.path.join(result_dir, "complete_results.txt")
    
    if not os.path.exists(feature_file):
        print(f"Warning: Feature list file {feature_file} not found")
        # Fallback to all features in the dataset
        try:
            df = pd.read_csv("data/pheme/pheme_paper_features.csv")
            return [col for col in df.columns if col not in ['source', 'label']]
        except FileNotFoundError:
            print("Warning: Feature file not found. This will cause errors during classification.")
            return []
    
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
    
    # Fallback to trying to get features from the model itself
    print("Couldn't get features from complete_results.txt, using fallback method.")
    return []

feature_names = get_model_features()

@app.route('/')
def home():
    return '''
    <h1>Fake News Classification API</h1>
    <p>Use POST /classify with a CSV file and row index to get classification.</p>
    '''

@app.route('/classify', methods=['POST'])
def classify():
    try:
        # Check if model is loaded
        if 'model' not in globals() or model is None:
            return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500
            
        # Get the CSV file path and row index from the request
        data = request.get_json()
        if not data or 'csv_path' not in data or 'row_index' not in data:
            return jsonify({'error': 'Please provide csv_path and row_index'}), 400
        
        csv_path = data['csv_path']
        row_index = int(data['row_index'])
        
        # Security enhancement: Validate file path to prevent directory traversal attacks
        # Only allow files from the 'data' directory
        abs_path = os.path.abspath(csv_path)
        base_dir = os.path.abspath('data')
        
        if not abs_path.startswith(base_dir):
            return jsonify({'error': 'Access denied: Only files in the data directory are allowed'}), 403
        
        # Check if file exists
        if not os.path.exists(csv_path):
            return jsonify({'error': 'CSV file not found'}), 404
        
        # Load the CSV file
        df = pd.read_csv(csv_path)
        
        # Check if row index is valid
        if row_index < 0 or row_index >= len(df):
            return jsonify({'error': 'Invalid row index'}), 400
        
        # Extract features for the specified row
        row_data = df.iloc[row_index]
        
        # Check if all required features exist
        missing_features = [feat for feat in feature_names if feat not in df.columns]
        if missing_features:
            return jsonify({'error': f'Missing required features in CSV: {", ".join(missing_features)}'}), 400
            
        # Get only the features used by the model
        features = row_data[feature_names].fillna(df[feature_names].median())
        
        # Convert to numpy array
        X = np.array([features.to_numpy()])
        
        # Make prediction
        prediction = model.predict_proba(X)[0]
        
        # Prepare response
        result = {
            'row_index': row_index,
            'probability_fake': float(prediction[1]),
            'probability_real': float(prediction[0]),
            'classification': 'FAKE' if prediction[1] > 0.5 else 'REAL',
            'confidence': float(max(prediction))
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)