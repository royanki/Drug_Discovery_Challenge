import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import precision_recall_curve, auc
from collections import defaultdict

def precision_at_k_with_hits(y_true, y_score, k=128):
    """Calculate precision@k and hits@k."""
    if k > len(y_score):
        k = len(y_score)
    idx = np.argsort(y_score)[::-1][:k]
    hit_count = y_true[idx].sum()
    return hit_count / k, int(hit_count)

def calculate_comprehensive_metrics(y_true, y_pred, y_proba):
    """Calculate all evaluation metrics."""
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall_curve, precision_curve)
    ppv_128, hits_128 = precision_at_k_with_hits(y_true, y_proba, k=128)
    ppv_256, hits_256 = precision_at_k_with_hits(y_true, y_proba, k=256)
    
    return {
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred),
        "AUC-ROC": roc_auc_score(y_true, y_proba),
        "PR AUC": pr_auc,
        "PPV@128": ppv_128,
        "Hits@128": hits_128,
        "PPV@256": ppv_256,
        "Hits@256": hits_256
    }

def print_metrics(metrics, title):
    """Print metrics in a formatted way."""
    print(f"\n{title}:")
    print("-" * 50)
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric:<12}: {value:.4f}")
        else:
            print(f"{metric:<12}: {value}")

def create_comparison_table(results_dict, metrics_to_compare=None):
    """Create a comparison table from multiple results."""
    if metrics_to_compare is None:
        metrics_to_compare = ['F1-Score', 'AUC-ROC', 'PR AUC', 'PPV@128', 'PPV@256', 'Precision', 'Recall']
    
    comparison_df = pd.DataFrame(index=results_dict.keys(), columns=metrics_to_compare)
    
    for name, result in results_dict.items():
        metrics = result['metrics'] if 'metrics' in result else result
        for metric in metrics_to_compare:
            if metric in metrics:
                comparison_df.loc[name, metric] = metrics[metric]
    
    return comparison_df.astype(float)