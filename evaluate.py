"""
Model evaluation script 
"""
import torch
import numpy as np
import os
import sys
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix, recall_score, precision_score, 
    accuracy_score, f1_score, roc_auc_score, 
    average_precision_score, log_loss
)

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from models.student_network import create_student_network
from utils.data_utils import ECGDataset, load_ptbxl_data, prepare_mi_classification_data


def evaluation_exactly_as_original(dataloader, net, device):
    """
    Evaluation function EXACTLY matching original evaluation() function
    """
    net.eval()
    
    predict = []
    label = []
    predscore = []

    for data in dataloader:
        inp, lab = data
        inp, lab = inp.type(torch.float).to(device), lab.type(torch.float).to(device)
        
        with torch.no_grad():
            # Forward pass - EXACTLY as original
            out = torch.sigmoid(net(inp[:, :, :2496])[0])  # Use [0] to get classification output
            
            # Get predictions - EXACTLY as original
            pred = np.round(out.cpu().detach().numpy())
            predscorei = np.array(out.cpu().numpy(), dtype=float)
            
            predscore.append(predscorei)
            predict.append(pred)
            label.append(lab.cpu().numpy())

    # Concatenate results - EXACTLY as original
    predict = np.concatenate(predict)
    label = np.concatenate(label)
    predscore = np.concatenate(predscore)

    # Calculate metrics - EXACTLY as original
    cnf_matrix = confusion_matrix(label, predict, normalize='true')
    
    sen = recall_score(label, predict, average='weighted')
    pre = precision_score(label, predict, average='weighted')
    acc = accuracy_score(label, predict)
    F1 = f1_score(label, predict, average='weighted')
    AUROC = roc_auc_score(label, predscore, average='weighted')
    prauc = average_precision_score(label, predscore, average='weighted')
    NLL = log_loss(label, predscore)

    # Print results - EXACTLY as original format
    print(acc, '\n', pre, '\n', sen, '\n', F1, '\n', AUROC, '\n', prauc, '\n', NLL)
    print('\n', cnf_matrix)
    
    return {
        'accuracy': acc,
        'precision': pre, 
        'recall': sen,
        'f1_score': F1,
        'auroc': AUROC,
        'prauc': prauc,
        'nll': NLL,
        'confusion_matrix': cnf_matrix
    }


def load_test_data_exactly_as_original(config):
    """Load test data EXACTLY as in original code"""
    # Load PTB-XL dataset
    Y = load_ptbxl_data(config.STUDENT_DATA_PATH)
    
    # Get test fold data - EXACTLY as original
    Y_test = Y[(Y.strat_fold == config.DATA_CONFIG['test_fold'])].diagnostic_superclass
    
    # Get available patient files from test directory
    arr = os.listdir(config.STUDENT_DATA_PATH + "test/")
    patients = [int(os.path.splitext(i)[0]) for i in arr]
    pat = np.array(np.unique(patients))
    
    # Extract labels for test patients
    labels = []
    for i in pat:
        if i in Y_test.index:
            labels.append(Y_test[i])
        else:
            labels.append([])  # Handle missing patients
    
    # Separate MI and Normal cases - EXACTLY as original
    mi_patients = []
    norm_patients = []
    
    for i, label in enumerate(labels):
        if 'MI' in label:
            mi_patients.append(pat[i])
        elif 'NORM' in label:
            norm_patients.append(pat[i])
    
    # Combine datasets - EXACTLY as original
    test_inputs = norm_patients + mi_patients
    test_labels = np.concatenate((
        np.zeros(len(norm_patients)), 
        np.ones(len(mi_patients))
    ))
    
    return test_inputs, test_labels


def create_test_dataloader(test_inputs, test_labels, data_path, batch_size=None):
    """Create test dataloader - EXACTLY as original"""
    if batch_size is None:
        batch_size = len(test_inputs)  # EXACTLY as original: batch_size = len(datt)
    
    test_dataset = ECGDataset(test_inputs, test_labels, data_path + "test/")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return test_loader


def main():
    """Main evaluation function"""
    # Load configuration
    config = Config()
    device = config.DEVICE
    
    print(f"Using device: {device}")
    
    # Load trained student model
    student_model = create_student_network(config).to(device)
    
    # Load model weights
    if os.path.exists(config.STUDENT_MODEL_PATH):
        print(f"Loading model from {config.STUDENT_MODEL_PATH}")
        student_model.load_state_dict(torch.load(config.STUDENT_MODEL_PATH, map_location=device))
    else:
        print("Error: Model checkpoint not found!")
        return
    
    # Load test data
    print("Loading test data...")
    test_inputs, test_labels = load_test_data_exactly_as_original(config)
    
    print(f"Test samples: {len(test_inputs)} "
          f"(MI: {sum(test_labels)}, Normal: {len(test_labels) - sum(test_labels)})")
    
    # Create test dataloader
    test_loader = create_test_dataloader(
        test_inputs, test_labels, 
        config.STUDENT_DATA_PATH,
        batch_size=len(test_inputs)  # EXACTLY as original
    )
    
    # Evaluate model
    print("\n" + "="*50)
    print("EVALUATION RESULTS (PTB-XL Test Set)")
    print("="*50)
    
    results = evaluation_exactly_as_original(test_loader, student_model, device)
    
    return results


if __name__ == "__main__":
    main()
