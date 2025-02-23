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

## Usage

### Python Scripts
- `dataset_pheme.py`: Process and analyze the PHEME dataset.
- `dataset_credbank.py`: Process the CREDBANK dataset.
- `dataset_buzzfeed.py`: Process the BuzzFeed dataset.
- `feature_elmination.py`: Script for feature elimination.
- `create_train_datasets.py`: Generate training datasets.
- `explore_credbank_structure.py`: Explore the structure of CREDBANK.
- `visualize_pheme.py`: Visualize aspects of the PHEME dataset.
- `app.py`: Flask application providing a REST API for fake news classification.

### API Usage

The project includes a Flask API (`app.py`) that provides fake news classification capabilities:
- Endpoint: `/classify` (POST)
- Input: JSON with `csv_path` and `row_index`
- Output: Classification result with probabilities and confidence score

### Jupyter Notebook
- `FakeNews.ipynb`: A comprehensive notebook that demonstrates the overall analysis workflow.

## Directory Structure

- `data/` - Contains the datasets.
- `utils/` - Utility modules used across the project.
- `output/` & `results/` - Directories for storing generated outputs and analysis results.
- Other scripts and notebooks relevant to various stages of the project.

## Contributing

Feel free to contribute by opening issues or submitting pull requests. Any improvements to documentation, scripts, or performance are welcome.
