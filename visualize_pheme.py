import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import os
import gc
from textwrap import wrap
import numpy as np
from dataset_pheme import load_pheme_dataset_raw

def preprocess_data(data):
    """Preprocess the dataset to handle NaN, infinite values, and other issues."""
    # Make a copy to avoid modifying the original data
    processed_data = data.copy()
    
    # Replace infinite values with NaN
    processed_data = processed_data.replace([np.inf, -np.inf], np.nan)
    
    # For numeric columns, handle NaN values
    numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        # Check if the column has any NaN values
        if processed_data[col].isna().any():
            # For count-based features (non-negative integers), replace NaN with 0
            if (processed_data[col].dtype in ['Int64', 'int64'] and 
                processed_data[col].dropna().min() >= 0):
                processed_data[col] = processed_data[col].fillna(0)
            # For other numeric features, replace NaN with median
            else:
                median_val = processed_data[col].dropna().median()
                if pd.isna(median_val):  # If median is also NaN
                    processed_data[col] = processed_data[col].fillna(0)
                else:
                    processed_data[col] = processed_data[col].fillna(median_val)
    
    # Handle division by zero for ratio-based features
    # Add a small epsilon to denominators that could be zero
    epsilon = 1e-10
    ratio_features = [col for col in numeric_cols if 'ratio' in col.lower()]
    for col in ratio_features:
        if col in processed_data.columns:
            processed_data[col] = processed_data[col].clip(-1e6, 1e6)  # Clip extreme values
            processed_data[col] = processed_data[col].replace([-np.inf, np.inf], np.nan)
            processed_data[col] = processed_data[col].fillna(processed_data[col].median())
    
    return processed_data

def create_plotly_visualizations(data, output_dir, dataset_type, data_state):
    """Create interactive Plotly visualizations for the dataset."""
    try:
        # Preprocess the data first
        processed_data = preprocess_data(data)
        
        if dataset_type not in ['paper_features', 'all_features']:
            # Interactive stacked bar chart for rumor distribution
            fig = px.bar(
                processed_data.groupby(['event', 'category']).size().reset_index(name='count'),
                x='event',
                y='count',
                color='category',
                title='Distribution of Rumours vs Non-rumours by Event',
                barmode='stack'
            )
            fig.write_html(os.path.join(output_dir, f'plotly_rumour_distribution_{dataset_type}.html'))

            # Timeline of tweets per event
            if 'tweet_timestamp' in processed_data.columns:
                processed_data['tweet_date'] = pd.to_datetime(processed_data['tweet_timestamp']).dt.date
                timeline_data = processed_data.groupby(['tweet_date', 'event', 'category']).size().reset_index(name='count')
                fig = px.line(
                    timeline_data,
                    x='tweet_date',
                    y='count',
                    color='event',
                    line_dash='category',
                    title='Timeline of Tweets by Event and Category'
                )
                fig.write_html(os.path.join(output_dir, f'plotly_tweet_timeline_{dataset_type}.html'))

            # Reaction analysis
            if 'num_reactions' in processed_data.columns and 'source_tweet_favorite_count' in processed_data.columns:
                fig = make_subplots(rows=1, cols=2, subplot_titles=(
                    'Reactions vs Favorites',
                    'Reactions vs Retweets'
                ))
                
                # Create scatter plots only for rows with valid data
                valid_mask = (
                    processed_data['num_reactions'].notna() & 
                    processed_data['source_tweet_favorite_count'].notna() & 
                    processed_data['source_tweet_retweet_count'].notna()
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=processed_data.loc[valid_mask, 'source_tweet_favorite_count'],
                        y=processed_data.loc[valid_mask, 'num_reactions'],
                        mode='markers',
                        marker=dict(color=processed_data.loc[valid_mask, 'category'].map({'rumour': 'red', 'non-rumour': 'blue'})),
                        name='Reactions vs Favorites'
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=processed_data.loc[valid_mask, 'source_tweet_retweet_count'],
                        y=processed_data.loc[valid_mask, 'num_reactions'],
                        mode='markers',
                        marker=dict(color=processed_data.loc[valid_mask, 'category'].map({'rumour': 'red', 'non-rumour': 'blue'})),
                        name='Reactions vs Retweets'
                    ),
                    row=1, col=2
                )
                
                fig.update_layout(title='Engagement Analysis', height=600)
                fig.write_html(os.path.join(output_dir, f'plotly_engagement_analysis_{dataset_type}.html'))
        else:
            # Feature correlation heatmap
            feature_cols = [col for col in processed_data.columns if col not in ['source', 'label']]
            
            # Calculate correlation matrix with numeric features only
            numeric_features = processed_data[feature_cols].select_dtypes(include=[np.number]).columns
            if len(numeric_features) > 0:
                # Remove features with all constant values
                valid_features = []
                for col in numeric_features:
                    if processed_data[col].nunique() > 1:
                        valid_features.append(col)
                
                if valid_features:
                    correlation_matrix = processed_data[valid_features].corr()
                    
                    # Replace any remaining NaN values in correlation matrix
                    correlation_matrix = correlation_matrix.fillna(0)
                    
                    fig = go.Figure(data=go.Heatmap(
                        z=correlation_matrix,
                        x=valid_features,
                        y=valid_features,
                        colorscale='RdBu',
                        zmid=0
                    ))
                    fig.update_layout(
                        title='Interactive Feature Correlation Matrix',
                        height=800,
                        width=800,
                        xaxis_tickangle=-45
                    )
                    fig.write_html(os.path.join(output_dir, f'plotly_feature_correlation_{dataset_type}.html'))

            # Feature distributions
            MAX_FEATURES_PER_FIGURE = 15
            numeric_features = list(processed_data[feature_cols].select_dtypes(include=[np.number]).columns)
            
            # Remove constant features and features with all NaN values
            valid_features = []
            for col in numeric_features:
                unique_vals = processed_data[col].nunique()
                if unique_vals > 1 and not processed_data[col].isna().all():
                    valid_features.append(col)
            
            for i in range(0, len(valid_features), MAX_FEATURES_PER_FIGURE):
                current_features = valid_features[i:i + MAX_FEATURES_PER_FIGURE]
                
                fig = make_subplots(
                    rows=len(current_features),
                    cols=1,
                    subplot_titles=current_features,
                    vertical_spacing=max(0.02, 1 / (len(current_features) * 4))
                )
                
                for j, col in enumerate(current_features, 1):
                    # Get statistics for the violin plot
                    data_series = processed_data[col].dropna()
                    
                    if len(data_series) > 0:
                        q1 = data_series.quantile(0.25)
                        q3 = data_series.quantile(0.75)
                        iqr = q3 - q1
                        lower_bound = q1 - 1.5 * iqr
                        upper_bound = q3 + 1.5 * iqr
                        
                        # Filter outliers for better visualization
                        plot_data = data_series.clip(lower_bound, upper_bound)
                        
                        fig.add_trace(
                            go.Violin(
                                y=plot_data,
                                name=col,
                                box_visible=True,
                                meanline_visible=True,
                                showlegend=False
                            ),
                            row=j, col=1
                        )
                
                fig.update_layout(
                    title=f'Interactive Feature Distributions (Group {i//MAX_FEATURES_PER_FIGURE + 1})',
                    height=250 * len(current_features),
                    showlegend=False
                )
                
                for j in range(len(current_features)):
                    fig.update_yaxes(title_text=current_features[j], row=j+1, col=1)
                
                fig.write_html(os.path.join(output_dir, f'plotly_feature_distributions_{dataset_type}_group_{i//MAX_FEATURES_PER_FIGURE + 1}.html'))

            # Feature importance if label column exists
            if 'label' in processed_data.columns:
                feature_importance = {}
                numeric_features = processed_data[feature_cols].select_dtypes(include=[np.number]).columns
                
                for col in numeric_features:
                    try:
                        # Skip constant features
                        if processed_data[col].nunique() <= 1:
                            continue
                            
                        correlation = abs(processed_data[col].corr(processed_data['label']))
                        if not pd.isna(correlation):
                            feature_importance[col] = correlation
                    except Exception as e:
                        print(f"Warning: Could not calculate correlation for feature '{col}': {str(e)}")
                        continue
                
                if feature_importance:
                    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                    
                    fig = go.Figure(go.Bar(
                        x=[x[0] for x in sorted_features],
                        y=[x[1] for x in sorted_features],
                        text=[f'{x[1]:.3f}' for x in sorted_features],
                        textposition='auto',
                    ))
                    
                    fig.update_layout(
                        title='Feature Importance (Based on Correlation with Label)',
                        xaxis_title='Features',
                        yaxis_title='Absolute Correlation with Label',
                        height=600,
                        xaxis_tickangle=-45
                    )
                    fig.write_html(os.path.join(output_dir, f'plotly_feature_importance_{dataset_type}.html'))
    
    except Exception as e:
        print(f"Error in create_plotly_visualizations: {str(e)}")
        raise

def visualize_pheme_dataset(file_path: str):
    """Visualize PHEME dataset information and create informative plots."""
    # Get absolute path relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create output directories for each dataset type
    dataset_types = ['raw', 'extended', 'paper_features', 'all_features']
    output_dirs = {}
    
    for dtype in dataset_types:
        output_dir = os.path.join(script_dir, 'output', 'pheme', dtype)
        os.makedirs(output_dir, exist_ok=True)
        output_dirs[dtype] = output_dir
        print(f"Created directory: {output_dir}")
    
    try:
        # Load the target dataset
        dtype_dict = {
            'source_tweet_favorite_count': 'Int64',
            'source_tweet_retweet_count': 'Int64',
            'num_reactions': 'Int64',
            'is_source': 'boolean'
        }
        
        data = pd.read_csv(file_path, low_memory=False, dtype=dtype_dict)
        
        if len(data) == 0:
            print("No data loaded")
            return
        
        print(f"Loaded {len(data):,} rows from {os.path.basename(file_path)}")
        
        # Determine dataset type from filename
        dataset_type = None
        filename = os.path.basename(file_path).lower()
        
        if 'raw_dataset' in filename:
            dataset_type = 'raw'
        elif 'extended_dataset' in filename:
            dataset_type = 'extended'
        elif 'paper_features' in filename:
            dataset_type = 'paper_features'
        elif 'all_features' in filename:
            dataset_type = 'all_features'
        else:
            print(f"Warning: Unknown dataset type from file path: {file_path}")
            dataset_type = 'unknown'
        
        print(f"Processing dataset type: {dataset_type}")
        
        # Create both static and interactive visualizations
        output_dir = output_dirs.get(dataset_type, output_dirs['raw'])
        create_visualizations(data, output_dir, dataset_type, dataset_type.replace('_', ' ').title())
        create_plotly_visualizations(data, output_dir, dataset_type, dataset_type.replace('_', ' ').title())
        
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
    finally:
        # Clear memory
        if 'data' in locals():
            del data
        gc.collect()

def create_visualizations(data, output_dir, dataset_type, data_state):
    """Create all visualizations for a given dataset."""
    # Figure 1: Dataset Overview
    plt.figure(figsize=(12, 6))
    
    # Prepare info text based on dataset type
    if dataset_type in ['paper_features', 'all_features']:
        info_text = (
            f"PHEME Dataset Overview ({data_state})\n\n"
            f"Total Rows: {len(data):,}\n"
            f"Columns: {len(data.columns)}\n"
            f"Memory Usage: {data.memory_usage().sum() / 1024**2:.2f} MB\n"
            f"Features: {', '.join([col for col in data.columns if col not in ['source', 'label']])}"
        )
    else:
        info_text = (
            f"PHEME Dataset Overview ({data_state})\n\n"
            f"Total Rows: {len(data):,}\n"
            f"Columns: {len(data.columns)}\n"
            f"Memory Usage: {data.memory_usage().sum() / 1024**2:.2f} MB\n"
            f"Features: {', '.join([col for col in data.columns])}"
            f"Events: {', '.join(data['event'].unique())}"
        )
    
    plt.text(0.5, 0.5, info_text, fontsize=12,
             horizontalalignment='center',
             verticalalignment='center',
             bbox=dict(facecolor='#f8f9fa', alpha=1.0,
                      edgecolor='#dee2e6', boxstyle='square,pad=1.5',
                      linewidth=2))
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, f'dataset_overview_{dataset_type}.png'), bbox_inches='tight', dpi=300)
    plt.close()

    # Only create event-based visualizations for raw and extended datasets
    if dataset_type not in ['paper_features', 'all_features']:
        # Figure 2: Distribution of Rumours vs Non-rumours by Event
        plt.figure(figsize=(12, 6))
        event_category_counts = data.groupby(['event', 'category']).size().unstack()
        event_category_counts.plot(kind='bar', stacked=True)
        plt.title('Distribution of Rumours vs Non-rumours by Event', fontsize=16)
        plt.xlabel('Event', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.legend(title='Category')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'rumour_distribution_{dataset_type}.png'), bbox_inches='tight', dpi=300)
        plt.close()

        # Figure 3: Reactions Distribution
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=data, x='event', y='num_reactions', hue='category')
        plt.title('Distribution of Reactions by Event and Category', fontsize=16)
        plt.xlabel('Event', fontsize=12)
        plt.ylabel('Number of Reactions', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'reactions_distribution_{dataset_type}.png'), bbox_inches='tight', dpi=300)
        plt.close()

        # Figure 4: Word Cloud of Source Tweets
        plt.figure(figsize=(12, 8))
        all_tweets = ' '.join(data['source_tweet_text'].dropna())
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_tweets)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud of Source Tweets', fontsize=16)
        plt.savefig(os.path.join(output_dir, f'wordcloud_source_tweets_{dataset_type}.png'), bbox_inches='tight', dpi=300)
        plt.close()
    else:
        # Figure 2: Feature Distribution
        plt.figure(figsize=(15, 8))
        feature_cols = [col for col in data.columns if col not in ['source', 'label']]
        feature_data = data[feature_cols]
        
        # Create boxplots for each feature
        plt.boxplot([feature_data[col].dropna() for col in feature_cols], tick_labels=feature_cols)
        plt.title('Distribution of Features', fontsize=16)
        plt.xticks(rotation=90)
        plt.ylabel('Value')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'feature_distribution_{dataset_type}.png'), bbox_inches='tight', dpi=300)
        plt.close()

        # Figure 3: Feature Correlation Matrix
        plt.figure(figsize=(12, 10))
        correlation_matrix = feature_data.corr()
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'feature_correlation_{dataset_type}.png'), bbox_inches='tight', dpi=300)
        plt.close()

    # Figure 5: Sample Entry
    plt.figure(figsize=(12, 12))
    sample_row = data.sample(n=1).iloc[0]
    
    # Prepare text for the sample
    sample_text = "Sample Entry:\n\n"
    
    # Show all columns except very large text fields
    for col in sample_row.index:
        if col not in ['source_tweet_text', 'reaction_texts']:
            value = str(sample_row[col])
            wrapped_value = '\n    '.join(wrap(value, width=80))
            sample_text += f"{col}:\n    {wrapped_value}\n\n"
    
    plt.text(0.5, 0.5, sample_text, fontsize=11,
             horizontalalignment='center',
             verticalalignment='center',
             bbox=dict(facecolor='#f8f9fa', alpha=1.0,
                      edgecolor='#dee2e6', boxstyle='square,pad=1.5',
                      linewidth=2))
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, f'sample_entry_{dataset_type}.png'), bbox_inches='tight', dpi=300)
    plt.close()

    # Clear memory after visualizations
    plt.close('all')
    gc.collect()

def main():
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define all PHEME dataset files with correct paths
    dataset_files = [
        os.path.join(script_dir, 'data', 'pheme', 'pheme_raw_dataset.csv'),
        os.path.join(script_dir, 'data', 'pheme', 'pheme_extended_dataset.csv'),
        os.path.join(script_dir, 'data', 'pheme', 'pheme_paper_features.csv'),
        os.path.join(script_dir, 'data', 'pheme', 'pheme_all_features.csv')
    ]
    
    # Process each file
    for file_path in dataset_files:
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                print(f"Error: File not found: {file_path}")
                continue
                
            print(f"\nProcessing: {file_path}")
            visualize_pheme_dataset(file_path)
            print(f"Successfully processed: {file_path}")
            
            # Force garbage collection
            plt.close('all')
            gc.collect()
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
        
        # Clear memory after each file
        gc.collect()

if __name__ == "__main__":
    main() 