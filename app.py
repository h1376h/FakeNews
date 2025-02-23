from flask import Flask, request, jsonify
from flask_ngrok import run_with_ngrok
import pandas as pd
import numpy as np
from joblib import load
import os

app = Flask(__name__)
run_with_ngrok(app)

# Load the trained model
model = load('results/elimination/best_classifier.joblib')

# Load feature names from the original dataset
def get_feature_names():
    # Using the same file path as in feature_elimination.py
    df = pd.read_csv("data/pheme/pheme_paper_features.csv")
    return [col for col in df.columns if col not in ['source', 'label']]

feature_names = get_feature_names()

@app.route('/')
def home():
    return '''
    <h1>Fake News Classification API</h1>
    <p>Use POST /classify with a CSV file and row index to get classification.</p>
    '''

@app.route('/classify', methods=['POST'])
def classify():
    try:
        # Get the CSV file path and row index from the request
        data = request.get_json()
        if not data or 'csv_path' not in data or 'row_index' not in data:
            return jsonify({'error': 'Please provide csv_path and row_index'}), 400
        
        csv_path = data['csv_path']
        row_index = int(data['row_index'])
        
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
        features = row_data[feature_names].fillna(df[feature_names].median())
        
        # Make prediction
        prediction = model.predict_proba([features])[0]
        
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
    app.run() 