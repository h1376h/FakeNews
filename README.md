# Fake News Detection and Analysis

This repository contains scripts and notebooks for analyzing and detecting fake news across multiple datasets. It includes data preprocessing, feature selection, visualization, and machine learning components tailored for fake news research.

## Datasets

Ensure you have the following datasets downloaded and placed in the appropriate directories:
- **PHEME Dataset:** Place the dataset in `data/pheme/pheme-rnr-dataset`.
- **CREDBANK Dataset:** Place the dataset in `data/credbank/CREDBANK`.
- **BuzzFeed Dataset:** Place the dataset in `data/buzzfeed`.

## Installation

Install the necessary dependencies using:

```bash
pip install -r requirements.txt
```

For Google Colab deployment, you'll also need to install `flask-ngrok`:

```bash
pip install flask-ngrok
```

## Usage

### Python Scripts
- `dataset_pheme.py`: Process and analyze the PHEME dataset.
- `dataset_credbank.py`: Process the CREDBANK dataset.
- `dataset_buzzfeed.py`: Process the BuzzFeed dataset.
- `feature_elimination.py`: Perform feature selection using recursive feature elimination.
- `create_train_datasets.py`: Generate training datasets from raw data.
- `explore_credbank_structure.py`: Explore the structure of CREDBANK dataset.
- `visualize_pheme.py`: Visualize aspects of the PHEME dataset.
- `examples.py`: Demonstrates how to use the fake news detection API in different ways.

### API Usage

The project includes two Flask application implementations:

1. **Local Deployment** (`app-local.py`):
   - Standard Flask application for local deployment
   - Run with: `python app-local.py`
   - Accessible at: `http://127.0.0.1:5000`

2. **Google Colab Deployment** (`app-google-colab.py`):
   - Uses flask-ngrok for exposing the API from Google Colab
   - Run in a Colab notebook with: `!python app-google-colab.py`
   - Generates a public URL for access

Both implementations provide:
- Endpoint: `/classify` (POST)
- Input: JSON with `csv_path` and `row_index`
- Output: Classification result with probabilities and confidence score

Example request:
```json
{
  "csv_path": "data/pheme/pheme_paper_features.csv",
  "row_index": 0
}
```

### Using Examples Script

The `examples.py` script provides three ways to test the fake news detection functionality:

1. **Local API Testing**: Tests the Flask API running on your local machine
   ```python
   # Make sure the Flask API is running (python app-local.py)
   python examples.py  # Uncomment the test_local_api() line first
   ```

2. **Direct Python Simulation**: Simulates the API response directly in Python
   ```python
   python examples.py  # Uses simulate_api_response() by default
   ```

3. **Google Colab API Testing**: Tests the API deployed on Google Colab via ngrok
   ```python
   # Update the ngrok URL in the script first
   python examples.py  # Uncomment the test_colab_api() line first
   ```

You can modify the CSV path and row index in the script to test different data points.

### Jupyter Notebook
- `FakeNews.ipynb`: A comprehensive notebook that demonstrates the overall analysis workflow.

## Directory Structure

- `data/` - Contains the datasets.
- `utils/` - Utility modules used across the project:
  - `features/` - Feature extraction modules for different datasets
  - `dataset_alignment.py` - Tools for aligning different dataset formats
- `output/` - Directory for storing intermediate processing outputs.
- `results/` - Directory for storing final analysis results and trained models.
- `nltk_data/` - NLTK data files for text processing.

## Feature Extraction Architecture

The project implements a modular feature extraction system:

- Base extractors in `utils/features/base*.py` files
- Dataset-specific extractors in `utils/features/[dataset]_[feature_type].py` files

Feature types include:
- Structural features (tweet structure, links, etc.)
- Content features (text analysis, sentiment, etc.)
- User features (author characteristics)
- Temporal features (timing patterns)

## Contributing

Feel free to contribute by opening issues or submitting pull requests. Any improvements to documentation, scripts, or performance are welcome. 