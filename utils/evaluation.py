"""
Evaluation utilities for ECG MI detection and delineation tasks
"""
import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    recall_score, precision_score, accuracy_score, f1_score, 
    roc_auc_score, average_precision_score, log_loss
)
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_mi_classification(model, test_loader, device, threshold=0.5):
    """
    Evaluate MI classification performance
    
    Args:
        model: Trained student model
        test_loader: Test data loader
        device: Device to run evaluation on
        threshold: Classification threshold
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    all_predictions = []
    all_labels = []
    all_scores = []
    
    with torch.no_grad():
        for batch_idx, (ecg_data, labels) in enumerate(test_loader):
            ecg_data = ecg_data.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)
            
            # Forward pass
            if hasattr(model, 'predict'):
                scores = model.predict(ecg_data[:, :, :2496])
            else:
                outputs, _ = model(ecg_data[:, :, :2496])
                scores = torch.sigmoid(outputs)
            
            predictions = (scores > threshold).float()
            
            all_predictions.extend(predictions.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
            all_scores.extend(scores.cpu().numpy().flatten())
    
    # Convert to numpy arrays
    predictions = np.array(all_predictions)
    labels = np.array(all_labels)
    scores = np.array(all_scores)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision_score(labels, predictions, average='weighted'),
        'recall': recall_score(labels, predictions, average='weighted'),
        'f1_score': f1_score(labels, predictions, average='weighted'),
        'auroc': roc_auc_score(labels, scores),
        'prauc': average_precision_score(labels, scores),
        'nll': log_loss(labels, scores),
        'confusion_matrix': confusion_matrix(labels, predictions, normalize='true')
    }
    
    return metrics, predictions, labels, scores


def plot_confusion_matrix(cm, class_names=['Normal', 'MI'], title='Confusion Matrix', save_path=None):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.3f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_roc_curve(labels, scores, title='ROC Curve', save_path=None):
    """Plot ROC curve"""
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_precision_recall_curve(labels, scores, title='Precision-Recall Curve', save_path=None):
    """Plot Precision-Recall curve"""
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    precision, recall, _ = precision_recall_curve(labels, scores)
    pr_auc = average_precision_score(labels, scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2,
             label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower left")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def print_classification_report(labels, predictions, class_names=['Normal', 'MI']):
    """Print detailed classification report"""
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    
    report = classification_report(labels, predictions, target_names=class_names)
    print(report)


def find_wave_onset(wave_category):
    """Find wave onset points"""
    onsets = []
    prev = 0
    for i, val in enumerate(wave_category):
        if val != 0 and prev == 0:
            onsets.append(i)
        prev = val
    return np.array(onsets)


def find_wave_offset(wave_category):
    """Find wave offset points"""
    offsets = []
    prev = 0
    for i, val in enumerate(wave_category):
        if val == 0 and prev != 0:
            offsets.append(i)
        prev = val
    return np.array(offsets)


def get_matching_timepoints(shorter, longer):
    """Get probably matching timepoints"""
    if len(shorter) == 0:
        return np.array([])
    
    indices = [
        np.argmin(row) for row in np.abs(np.subtract.outer(shorter, longer))
    ]
    return longer[indices]


def compute_tf_mismatch(shorter, longer, tolerance):
    """Compute true/false matches with tolerance"""
    if len(shorter) == 0:
        return len(longer), 0, []

    falses = len(longer) - len(shorter)
    matched_longer = get_matching_timepoints(shorter, longer)
    dists = np.abs(matched_longer - shorter)
    trues = np.sum(dists <= tolerance)
    errors = list(dists[dists <= tolerance])

    return falses, trues, errors


def pointwise_evaluation(test_points, pred_points, tolerance):
    """Pointwise evaluation for delineation"""
    tp, fn, fp, errors = [], [], [], []

    if len(test_points) == len(pred_points):
        _, trues, errs = compute_tf_mismatch(test_points, pred_points, tolerance)
        tp.append(trues)
        errors.extend(errs)

    elif len(test_points) > len(pred_points):
        falses, trues, errs = compute_tf_mismatch(pred_points, test_points, tolerance)
        tp.append(trues)
        fn.append(falses)
        errors.extend(errs)

    elif len(test_points) < len(pred_points):
        falses, trues, errs = compute_tf_mismatch(test_points, pred_points, tolerance)
        tp.append(trues)
        fp.append(falses)
        errors.extend(errs)

    return tp, fp, fn, errors


def evaluate_delineation(y_true, y_pred, tolerance=10):
    """
    Evaluate ECG delineation performance
    
    Args:
        y_true: True segmentation masks
        y_pred: Predicted segmentation masks
        tolerance: Tolerance for onset/offset detection (in samples)
    
    Returns:
        Dictionary with delineation metrics
    """
    metrics_onset = []
    metrics_offset = []
    
    for i in range(len(y_true)):
        # Find onset and offset points
        true_onsets = find_wave_onset(y_true[i])
        pred_onsets = find_wave_onset(y_pred[i])
        
        true_offsets = find_wave_offset(y_true[i])
        pred_offsets = find_wave_offset(y_pred[i])
        
        # Evaluate onsets and offsets
        onset_metrics = pointwise_evaluation(true_onsets, pred_onsets, tolerance)
        offset_metrics = pointwise_evaluation(true_offsets, pred_offsets, tolerance)
        
        metrics_onset.append(onset_metrics)
        metrics_offset.append(offset_metrics)
    
    # Aggregate metrics
    def aggregate_metrics(metrics_list):
        tp = sum(sum(m[0]) for m in metrics_list)
        fp = sum(sum(m[1]) for m in metrics_list)
        fn = sum(sum(m[2]) for m in metrics_list)
        errors = [err for m in metrics_list for err in m[3]]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        
        error_mean = np.mean(errors) if errors else 0
        error_std = np.std(errors) if errors else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'error_mean': error_mean,
            'error_std': error_std
        }
    
    onset_results = aggregate_metrics(metrics_onset)
    offset_results = aggregate_metrics(metrics_offset)
    
    return {
        'onset': onset_results,
        'offset': offset_results
    }


def print_delineation_results(results, wave_name='Wave'):
    """Print delineation evaluation results"""
    print(f"\n{wave_name} Delineation Results:")
    print("-" * 40)
    print(f"Onset  - Precision: {results['onset']['precision']:.4f}, "
          f"Recall: {results['onset']['recall']:.4f}, "
          f"F1: {results['onset']['f1']:.4f}")
    print(f"         Error: {results['onset']['error_mean']:.2f} ± {results['onset']['error_std']:.2f} samples")
    
    print(f"Offset - Precision: {results['offset']['precision']:.4f}, "
          f"Recall: {results['offset']['recall']:.4f}, "
          f"F1: {results['offset']['f1']:.4f}")
    print(f"         Error: {results['offset']['error_mean']:.2f} ± {results['offset']['error_std']:.2f} samples")


def comprehensive_evaluation(model, test_loader, device, save_dir=None):
    """
    Comprehensive evaluation of the model
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run evaluation on
        save_dir: Directory to save plots
    
    Returns:
        Dictionary with all evaluation results
    """
    print("Starting comprehensive evaluation...")
    
    # Evaluate MI classification
    metrics, predictions, labels, scores = evaluate_mi_classification(model, test_loader, device)
    
    # Print results
    print("\n" + "="*60)
    print("MI CLASSIFICATION RESULTS")
    print("="*60)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print(f"AUROC:     {metrics['auroc']:.4f}")
    print(f"PRAUC:     {metrics['prauc']:.4f}")
    print(f"NLL:       {metrics['nll']:.4f}")
    
    # Print classification report
    print_classification_report(labels, predictions)
    
    # Plot visualizations
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Confusion matrix
        plot_confusion_matrix(
            metrics['confusion_matrix'], 
            title='MI Classification Confusion Matrix',
            save_path=os.path.join(save_dir, 'confusion_matrix.png')
        )
        
        # ROC curve
        plot_roc_curve(
            labels, scores,
            title='MI Classification ROC Curve',
            save_path=os.path.join(save_dir, 'roc_curve.png')
        )
        
        # PR curve
        plot_precision_recall_curve(
            labels, scores,
            title='MI Classification PR Curve',
            save_path=os.path.join(save_dir, 'pr_curve.png')
        )
    else:
        plot_confusion_matrix(metrics['confusion_matrix'])
        plot_roc_curve(labels, scores)
        plot_precision_recall_curve(labels, scores)
    
    return {
        'classification_metrics': metrics,
        'predictions': predictions,
        'labels': labels,
        'scores': scores
    }
