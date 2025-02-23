import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple
import os
from joblib import dump

def load_dataset(file_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load and preprocess the dataset."""
    df = pd.read_csv(file_path)
    feature_cols = [col for col in df.columns if col not in ['source', 'label']]
    X = df[feature_cols].fillna(df[feature_cols].median()).values
    y = df['label'].values
    return X, y, feature_cols

def evaluate_feature_set(X: np.ndarray, y: np.ndarray, clf) -> float:
    """Evaluate feature set using 5-fold cross-validation."""
    # Use StratifiedKFold to maintain class distribution
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    return scores.mean()

def recursive_feature_elimination(X: np.ndarray, y: np.ndarray, feature_names: List[str], 
                               output_dir: str) -> None:
    """Perform recursive feature elimination with early stopping but continue afterwards."""
    os.makedirs(output_dir, exist_ok=True)
    n_features = X.shape[1]
    
    # Initialize results tracking
    results = {
        'n_features': [],
        'removed_feature': [],
        'roc_auc': [],
        'remaining_features': [],
        'roc_auc_std': [],
        'phase': []  # Track whether feature was removed before or after early stopping
    }
    
    # Initialize classifier with fixed parameters
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    
    # Keep track of current feature set
    current_features = list(range(n_features))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    initial_scores = cross_val_score(clf, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    best_score = initial_scores.mean()
    best_score_std = initial_scores.std()
    best_feature_set = current_features.copy()
    best_classifier = None
    
    # Save initial state
    clf.fit(X, y)
    dump(clf, f"{output_dir}/initial_classifier.joblib")
    
    print(f"Initial ROC-AUC with {n_features} features: {best_score:.4f} ± {best_score_std:.4f}")
    
    early_stopping_triggered = False
    early_stopping_features = None
    
    # Main elimination loop
    while len(current_features) > 1:
        scores = []
        # Try removing each feature
        for idx in current_features:
            features_without_current = [f for f in current_features if f != idx]
            cv_scores = cross_val_score(clf, X[:, features_without_current], y, cv=cv, scoring='roc_auc', n_jobs=-1)
            scores.append((cv_scores.mean(), cv_scores.std(), idx))
        
        # Find feature whose removal gives best score
        best_iteration_score, best_iteration_std, worst_feature = max(scores, key=lambda x: x[0])
        
        # Remove the feature
        current_features.remove(worst_feature)
        
        # Store results
        results['n_features'].append(len(current_features))
        results['removed_feature'].append(feature_names[worst_feature])
        results['roc_auc'].append(best_iteration_score)
        results['roc_auc_std'].append(best_iteration_std)
        results['remaining_features'].append([feature_names[i] for i in current_features])
        results['phase'].append('pre_stopping' if not early_stopping_triggered else 'post_stopping')
        
        print(f"Removed: {feature_names[worst_feature]} | ROC-AUC: {best_iteration_score:.4f} ± {best_iteration_std:.4f} | Remaining: {len(current_features)}")
        
        # Update best score and feature set if improved
        if best_iteration_score > best_score:
            best_score = best_iteration_score
            best_score_std = best_iteration_std
            best_feature_set = current_features.copy()
            # Train and save the best classifier
            clf.fit(X[:, best_feature_set], y)
            best_classifier = clf
            dump(clf, f"{output_dir}/best_classifier.joblib")
        
        # Early stopping if performance drops significantly
        if best_iteration_score < best_score - 0.01:
            print("Early stopping point reached - saving state")
            
            # Save early stopping state
            with open(f"{output_dir}/early_stopping_state.txt", 'w') as f:
                f.write(f"Early stopping triggered at {len(current_features)} features\n")
                f.write(f"ROC-AUC at stopping: {best_iteration_score:.4f} ± {best_iteration_std:.4f}\n")
                f.write("\nRemaining features at early stopping:\n")
                f.write('\n'.join([feature_names[i] for i in current_features]))
            
            # Save classifier at early stopping point
            clf_at_stopping = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
            clf_at_stopping.fit(X[:, current_features], y)
            dump(clf_at_stopping, f"{output_dir}/early_stopping_classifier.joblib")
            
            # Create plot at early stopping point
            plt.figure(figsize=(12, 6))
            plt.errorbar(results['n_features'], results['roc_auc'], 
                        yerr=results['roc_auc_std'], fmt='o-', capsize=5)
            plt.axvline(x=len(current_features), color='r', linestyle='--', label='Early Stopping Point')
            plt.xlabel('Number of Features')
            plt.ylabel('ROC-AUC Score')
            plt.title('Feature Elimination Performance at Early Stopping')
            plt.grid(True)
            plt.legend()
            plt.savefig(f"{output_dir}/early_stopping_plot.png")
            plt.close()
            
            # Continue with feature elimination instead of breaking
            early_stopping_triggered = True
            early_stopping_features = current_features.copy()
    
    # Save final results
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{output_dir}/elimination_results.csv", index=False)
    
    # Plot 1: ROC-AUC vs Number of Features with confidence intervals and early stopping point
    plt.figure(figsize=(12, 6))
    plt.errorbar(results['n_features'], results['roc_auc'], 
                yerr=results['roc_auc_std'], fmt='o-', capsize=5)
    if early_stopping_features:
        plt.axvline(x=len(early_stopping_features), color='r', linestyle='--', label='Early Stopping Point')
    plt.xlabel('Number of Features')
    plt.ylabel('ROC-AUC Score')
    plt.title('Complete Feature Elimination Performance')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{output_dir}/complete_elimination_plot.png")
    plt.close()
    
    # Plot 1b: Paper-style plot (similar to Figure 1)
    plt.figure(figsize=(7, 6))
    plt.plot(range(len(results['roc_auc'])-1, -1, -1), results['roc_auc'], '-', color='blue', linewidth=1.5, label='Model Performance')
    
    # Configure grid
    plt.grid(True, linestyle=':', color='gray', alpha=0.5)
    plt.gca().set_axisbelow(True)  # Put grid behind the plot
    
    # Configure axes
    plt.xlabel('Deleted Feature Count')
    plt.ylabel('ROC-AUC')
    
    # Set precise limits and ticks
    plt.ylim(0.62, 0.76)
    plt.xlim(0, 45)
    plt.yticks(np.arange(0.62, 0.77, 0.02))
    plt.xticks(np.arange(0, 46, 5))
    
    # Remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # Adjust legend
    plt.legend(loc='upper right', frameon=False)
    
    # Save with high DPI and tight layout
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_elimination_paper_style.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Feature Removal Impact
    plt.figure(figsize=(15, 8))
    removal_impact = pd.DataFrame({
        'Feature': results['removed_feature'],
        'ROC-AUC': results['roc_auc'],
        'Phase': results['phase']
    })
    sns.barplot(data=removal_impact, x='Feature', y='ROC-AUC', hue='Phase')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_removal_impact_with_phases.png")
    plt.close()
    
    # Plot 3: Feature Importance of Best Model
    if best_classifier is not None:
        plt.figure(figsize=(12, 6))
        importances = pd.DataFrame({
            'Feature': [feature_names[i] for i in best_feature_set],
            'Importance': best_classifier.feature_importances_
        })
        importances = importances.sort_values('Importance', ascending=False)
        sns.barplot(data=importances, x='Feature', y='Importance')
        plt.xticks(rotation=45, ha='right')
        plt.title('Feature Importance in Best Model')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/feature_importance.png")
        plt.close()
    
    # Save comprehensive results
    with open(f"{output_dir}/complete_results.txt", 'w') as f:
        f.write("=== Feature Elimination Results ===\n\n")
        
        f.write("Initial State:\n")
        f.write(f"- Features: {len(feature_names)}\n")
        f.write(f"- Initial ROC-AUC: {np.mean(initial_scores):.4f} ± {np.std(initial_scores):.4f}\n\n")
        
        if early_stopping_features:
            f.write("Early Stopping Point:\n")
            f.write(f"- Features remaining: {len(early_stopping_features)}\n")
            f.write("- Features at early stopping:\n")
            f.write('\n'.join([feature_names[i] for i in early_stopping_features]) + '\n\n')
        
        f.write("Best Model:\n")
        f.write(f"- Number of features: {len(best_feature_set)}\n")
        f.write(f"- Best ROC-AUC: {best_score:.4f} ± {best_score_std:.4f}\n")
        f.write("- Best feature set:\n")
        f.write('\n'.join([feature_names[i] for i in best_feature_set]) + '\n\n')
        
        f.write("Feature Importance in Best Model:\n")
        if best_classifier is not None:
            for feat, imp in zip([feature_names[i] for i in best_feature_set], 
                               best_classifier.feature_importances_):
                f.write(f"{feat}: {imp:.4f}\n")
        
        f.write("\nFinal State:\n")
        f.write(f"- Features remaining: {len(current_features)}\n")
        f.write("- Final features:\n")
        f.write('\n'.join([feature_names[i] for i in current_features]))

def main():
    """Main function to run feature elimination."""
    print("Starting feature elimination...")
    # input_file = "data/train/train_paper.csv"
    input_file = "data/pheme/pheme_paper_features.csv"
    output_dir = "results/elimination"
    X, y, feature_names = load_dataset(input_file)
    print(f"Loaded dataset with {X.shape[0]} samples and {X.shape[1]} features")
    recursive_feature_elimination(X, y, feature_names, output_dir)

if __name__ == "__main__":
    main()