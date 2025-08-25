"""
Teacher Network Evaluation Script - ECG Delineation Performance

"""
import torch
import numpy as np
import os
import sys
from torch.utils.data import TensorDataset, DataLoader
import neurokit2 as nk

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from models.teacher_network import create_teacher_network


def load_test_data_exactly_as_original():
    """Load QT database test data"""
    # Load test data 
    testset1 = np.load(#)
    testset2 = np.load(#)
    
    # Remove quality-flagged samples
    testsetql1 = np.delete(testset1, (#), axis=0)
    testsetql2 = np.delete(testset2, (#), axis=0)
    testset = np.concatenate((testsetql1, testsetql2))
    
    return testset


def minmax_normalize(x):
    """Normalize data EXACTLY as original minmax() function"""
    x = (x - np.mean(x)) / np.std(x)
    return x


def prepare_test_data(testset):
    """Prepare test data """
    xt = testset[:, 0, :]  # ECG signals
    for i in range(len(xt)):
        xt[i] = minmax_normalize(xt[i])

    yt = testset[:, 1:4, :]  # Segmentation masks: P->QRS->T

    # Convert to tensors
    test_inputs = torch.tensor(xt, dtype=torch.float32)    # Nx2500
    test_labels = torch.tensor(yt, dtype=torch.float32)    # Nx3x2500

    return test_inputs, test_labels


def apply_quality_control(test_inputs, test_labels):
    """Apply ECG quality control """
    dele = []
    for i in range(len(test_inputs)):
        ql = nk.ecg_quality(test_inputs[i].numpy(), rpeaks=None, sampling_rate=250, 
                           method="zhao2018", approach='simple')
        
        if ql == 'Unacceptable':
            dele.append(i)

    # Remove unacceptable quality signals
    test_inputs_clean = np.delete(test_inputs.numpy(), dele, axis=0)
    test_labels_clean = np.delete(test_labels.numpy(), dele, axis=0)
    
    print(f"Removed {len(dele)} low-quality signals")
    print(f"Remaining samples: {len(test_inputs_clean)}")
    
    return torch.tensor(test_inputs_clean), torch.tensor(test_labels_clean)


def find_wave_onset(wave_category):
    """Find wave onset points EXACTLY as original"""
    onsets = []
    prev = None
    for i, val in enumerate(wave_category):
        if val != 0 and prev == 0:
            onsets.append(i)
        prev = val
    return np.array(onsets)


def find_wave_offset(wave_category):
    """Find wave offset points """
    offsets = []
    prev = None
    for i, val in enumerate(wave_category):
        if val == 0 and prev != 0:
            offsets.append(i)
        prev = val
    return np.array(offsets)


def get_probably_matching_timepoints(shorter, longer):
    """Get matching timepoints """
    indices_minimizing_distances = [
        np.argmin(row) for row in np.abs(np.subtract.outer(shorter, longer))
    ]
    return longer[indices_minimizing_distances]


def TF_mismatch(shorter, longer, tolerance):
    """Compute true/false matches """
    if len(shorter) == 0:
        return len(longer), 0, []

    falses = len(longer) - len(shorter)
    matched_longer = get_probably_matching_timepoints(shorter, longer)
    dists = np.abs(matched_longer - shorter)
    trues = np.sum(dists <= tolerance)
    errs = list(dists[dists <= tolerance])

    return falses, trues, errs


def pointwise_evaluation(testpt, predpt, tolerance):
    """Pointwise evaluation """
    TP, FN, FP, error = [], [], [], []

    if len(testpt) == len(predpt):
        # Wave lengths match - check tolerance
        _, trues, errs = TF_mismatch(testpt, predpt, tolerance)
        TP.append(trues)
        error.append(errs)

    elif len(testpt) > len(predpt):
        # More ground truth waves - false negatives
        falses, trues, errs = TF_mismatch(predpt, testpt, tolerance)
        TP.append(trues)
        FN.append(falses)
        error.append(errs)

    elif len(testpt) < len(predpt):
        # More predicted waves - false positives
        falses, trues, errs = TF_mismatch(testpt, predpt, tolerance)
        TP.append(trues)
        FP.append(falses)
        error.append(errs)

    return TP, FP, FN, error


def final_evaluation_exactly_as_original(y_true, y_pred, tolerance):
    """Final evaluation """
    metrics_on = []
    metrics_off = []
    
    for i in range(len(y_true)):
        # Find onset and offset points
        y_hat_on = find_wave_onset(y_pred[i])
        y_on = find_wave_onset(y_true[i])

        y_hat_off = find_wave_offset(y_pred[i])
        y_off = find_wave_offset(y_true[i])

        # Evaluate onsets and offsets
        metrics_on.append(pointwise_evaluation(y_on, y_hat_on, tolerance))
        metrics_off.append(pointwise_evaluation(y_off, y_hat_off, tolerance))

    # Aggregate onset metrics
    TP_on = sum(np.sum(metrics_on, axis=0)[0])
    FP_on = sum(np.sum(metrics_on, axis=0)[1])
    FN_on = sum(np.sum(metrics_on, axis=0)[2])
    
    error_on_concat = np.concatenate(np.sum(metrics_on, axis=0)[3])
    error_on_mean = error_on_concat.mean() if len(error_on_concat) > 0 else 0
    error_on_std = error_on_concat.std() if len(error_on_concat) > 0 else 0

    # Aggregate offset metrics
    TP_off = sum(np.sum(metrics_off, axis=0)[0])
    FP_off = sum(np.sum(metrics_off, axis=0)[1])
    FN_off = sum(np.sum(metrics_off, axis=0)[2])
    
    error_off_concat = np.concatenate(np.sum(metrics_off, axis=0)[3])
    error_off_mean = error_off_concat.mean() if len(error_off_concat) > 0 else 0
    error_off_std = error_off_concat.std() if len(error_off_concat) > 0 else 0

    # Calculate metrics
    rec_on = TP_on / (TP_on + FN_on) if (TP_on + FN_on) > 0 else 0
    prec_on = TP_on / (TP_on + FP_on) if (TP_on + FP_on) > 0 else 0
    f1_on = 2 * TP_on / (2 * TP_on + FN_on + FP_on) if (2 * TP_on + FN_on + FP_on) > 0 else 0

    rec_off = TP_off / (TP_off + FN_off) if (TP_off + FN_off) > 0 else 0
    prec_off = TP_off / (TP_off + FP_off) if (TP_off + FP_off) > 0 else 0
    f1_off = 2 * TP_off / (2 * TP_off + FN_off + FP_off) if (2 * TP_off + FN_off + FP_off) > 0 else 0

    # Print results EXACTLY as original format
    print("Onset")
    print("recall:", rec_on)
    print("precision:", prec_on)
    print("F1:", f1_on)
    print("error:", error_on_mean)
    print("error_std:", error_on_std)

    print("\nOffset")
    print("recall:", rec_off)
    print("precision:", prec_off)
    print("f1:", f1_off)
    print("error:", error_off_mean)
    print("error_std:", error_off_std)

    return {
        'onset': {
            'recall': rec_on,
            'precision': prec_on,
            'f1': f1_on,
            'error_mean': error_on_mean,
            'error_std': error_on_std
        },
        'offset': {
            'recall': rec_off,
            'precision': prec_off,
            'f1': f1_off,
            'error_mean': error_off_mean,
            'error_std': error_off_std
        }
    }


def evaluate_teacher_model(model, test_inputs, test_labels, device, wave_idx=2, tolerance=2.5):
    """
    
    Args:
        model: Trained teacher model
        test_inputs: Test ECG signals
        test_labels: Ground truth segmentation masks
        device: Device for computation
        wave_idx: Which wave to evaluate (0=P, 1=QRS, 2=T-wave)
        tolerance: Tolerance for onset/offset detection
    """
    model.eval()
    
    y_true = []
    y_pred = []
    
    # Create dataloader
    batch_size = 32
    test_dataset = TensorDataset(test_inputs, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Evaluating teacher model on wave {wave_idx} (0=P, 1=QRS, 2=T-wave)")
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            x = x[:, :2496].to(device, dtype=torch.float)
            x = x.unsqueeze(1)  # Add channel dimension
            y = y[:, :3, :2496].to(device, dtype=torch.float)
            
            # Forward pass - get segmentation output
            pred, _, _, _ = model(x)  # Get d1, x1, x2, x3
            
            # Apply sigmoid and round for binary segmentation
            mask = torch.round(torch.sigmoid(pred))
            
            # Extract specific wave type
            y_true.append(y[:, wave_idx, :].cpu().numpy())
            y_pred.append(mask[:, wave_idx, :].cpu().numpy())
    
    # Concatenate all batches
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    
    print(f"Evaluating {len(y_true)} samples with tolerance {tolerance}")
    
    # Run evaluation EXACTLY as original
    results = final_evaluation_exactly_as_original(y_true, y_pred, tolerance)
    
    return results


def main():
    """Main evaluation function for teacher network"""
    # Load configuration
    config = Config()
    device = config.DEVICE
    
    print(f"Using device: {device}")
    
    # Load teacher model
    teacher_model = create_teacher_network(config).to(device)
    
    # Load model weights
    if os.path.exists(config.TEACHER_MODEL_PATH):
        print(f"Loading teacher model from {config.TEACHER_MODEL_PATH}")
        teacher_model.load_state_dict(torch.load(config.TEACHER_MODEL_PATH, map_location=device))
    else:
        print("Error: Teacher model checkpoint not found!")
        return
    
    # Load test data
    print("Loading QT database test data...")
    testset = load_test_data_exactly_as_original()
    test_inputs, test_labels = prepare_test_data(testset)
    
    print(f"Initial test samples: {len(test_inputs)}")
    
    # Apply quality control
    test_inputs, test_labels = apply_quality_control(test_inputs, test_labels)
    
    # Evaluate different wave types and tolerances
    wave_names = ['P-wave', 'QRS complex', 'T-wave']
    tolerances = [2.5, 6.25, 10, 17.5]  # EXACTLY as original comment: #2.5#6.25 #10 #17.5,37.5
    
    for wave_idx in range(3):  # P, QRS, T
        print(f"\n{'='*60}")
        print(f"EVALUATING {wave_names[wave_idx].upper()}")
        print('='*60)
        
        for tol in tolerances:
            print(f"\n--- Tolerance: {tol} samples ---")
            results = evaluate_teacher_model(
                teacher_model, test_inputs, test_labels, device, 
                wave_idx=wave_idx, tolerance=tol
            )
    
    print(f"\n{'='*60}")
    print("TEACHER NETWORK EVALUATION COMPLETED")
    print('='*60)


if __name__ == "__main__":
    main()
