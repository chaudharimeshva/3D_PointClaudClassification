# utils/metrics.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

def compute_metrics(y_true, y_pred, class_names=None):
    """Compute comprehensive classification metrics"""
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Macro and weighted averages
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'per_class_precision': precision,
        'per_class_recall': recall,
        'per_class_f1': f1,
        'per_class_support': support,
        'confusion_matrix': cm
    }
    
    return metrics

def plot_confusion_matrix(cm, class_names=None, save_path=None, figsize=(10, 8)):
    """Plot confusion matrix"""
    plt.figure(figsize=figsize)
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(cm))]
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap
    sns.heatmap(cm_normalized, 
                annot=True, 
                fmt='.2f', 
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    
    plt.title('Confusion Matrix (Normalized)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return plt.gcf()

def print_classification_report(y_true, y_pred, class_names=None):
    """Print detailed classification report"""
    if class_names is None:
        target_names = None
    else:
        target_names = class_names
    
    report = classification_report(y_true, y_pred, target_names=target_names)
    print("Classification Report:")
    print("-" * 50)
    print(report)
    
    return report