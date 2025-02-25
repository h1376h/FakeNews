# Dataset Alignment

This module implements the dataset alignment process described in the paper for fake news detection across multiple datasets.

## Overview

The alignment process works across three different datasets:
1. **BuzzFeed** (originally Facebook-based)
2. **CREDBANK** (Twitter-based)
3. **PHEME** (Twitter thread-based)

The goal is to generate a consistent feature set and labels across these datasets to enable fair comparison of models.

## Alignment Process

### 1. BuzzFeed Twitter Thread Extraction
- Extracts the 10 most shared stories from left-wing Facebook pages
- Extracts the 10 most shared stories from right-wing Facebook pages
- Searches Twitter for these headlines
- Keeps the top 3 most retweeted posts for each headline
- Results in ~35 topics with journalist-provided labels (15 "mostly true", 20 "mostly false")

### 2. CREDBANK Label Alignment
- The grand mean of CREDBANK's accuracy assessments is 1.7
- The median is 1.767
- The 25th and 75th quartiles are 1.6 and 1.867 respectively
- Events below the 15% quantile (mean rating < 1.467) become negative samples
- Events above the 85% quantile (mean rating > 1.9) become positive samples
- Events between these values are left unlabeled and removed from the dataset

### 3. Twitter Thread Structure Capture
- For CREDBANK: Identify the most retweeted tweet in each event as the thread root
- For BuzzFeed: Use popular headline tweets as thread roots
- Capture replies to construct thread structure similar to PHEME
- Discard CREDBANK threads with no reactions

## Usage

### Running the Full Alignment Process

```python
python dataset_alignment.py --base-path data --output-dir data/aligned
```

### Options
- `--base-path`: Base directory containing all datasets (default: 'data')
- `--output-dir`: Directory to save the output files (default: 'data/aligned')
- `--no-save`: Don't save intermediate CSV files

### Testing the Alignment Process

```python
python test_dataset_alignment.py --base-path data --output-dir data/aligned_test --test-type both
```

### Test Options
- `--test-type`: Type of test to run (full, steps, or both - default: both)
  - `full`: Run the complete alignment process end-to-end
  - `steps`: Test each step individually
  - `both`: Run both tests

## Data Directory Structure

The expected data directory structure is:

```
data/
├── buzzfeed/
│   └── (BuzzFeed dataset files)
├── credbank/
│   └── CREDBANK/
│       ├── cred_event_TurkRatings.data
│       ├── eventNonEvent_annotations.data
│       └── cred_event_SearchTweets.data
├── pheme/
│   └── pheme-rnr-dataset/
│       ├── rumours/
│       └── non-rumours/
└── aligned/
    └── (output files)
```

## Requirements

- Python 3.6+
- pandas
- numpy
- tqdm
- TextBlob (for sentiment analysis)
- Twitter API access (optional, fallback to mock data if not available)

## Outputs

The alignment process produces:
1. **Raw aligned datasets**: CSV files with the aligned data
2. **Feature datasets**: CSV files with extracted features
   - Paper features: Features used in the original paper
   - All features: Extended set of features for experimentation 