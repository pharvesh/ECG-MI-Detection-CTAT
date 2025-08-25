"""
Training script for Teacher Network (ECG Delineation) 
"""
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import os
import sys
import wfdb  # Added missing import

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from models.teacher_network import create_teacher_network
from utils.data_utils import (
    load_and_preprocess_teacher_data, apply_augmentation_exactly_as_original
)
from models.ctat_module import EarlyStopping


def train_teacher_model_exactly_as_original(model, train_dataloader, validation_dataloader, config, device):
    """Train teacher model """
    
    # Setup optimizer EXACTLY as original
    optimiser = SGD(
        params=model.parameters(), 
        lr=config.TEACHER_CONFIG['learning_rate'],  # 0.001
        momentum=config.TEACHER_CONFIG['momentum'],  # 0.9
        weight_decay=config.TEACHER_CONFIG['weight_decay']  # 1e-5
    )
    
    # Loss and scheduler EXACTLY as original
    criterion = nn.BCEWithLogitsLoss().to(device)
    scheduler = ReduceLROnPlateau(optimiser, 'min', factor=0.5, patience=10, verbose=True)
    
    # Early stopping EXACTLY as original
    early_stopping = EarlyStopping(
        patience=config.TEACHER_CONFIG['patience'],  # 40
        verbose=True,
        path=os.path.join(config.CHECKPOINT_DIR, 'seg_SGD_best.pt')
    )
    
    # Training history
    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []
    
    print("Starting teacher network training...")
    
    for epoch in trange(1, config.TEACHER_CONFIG['max_epochs'] + 1):  # 400 epochs
        
       
        model.train()  # prep model for training
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            ecg, seg = batch

            ecg = ecg[:, :2496].to(device=device, dtype=torch.float)  # EXACTLY as original
            ecg = ecg[:, np.newaxis, :]  # Add channel dimension
            seg = seg[:, :3, :2496].float().to(device)  # EXACTLY as original

            optimiser.zero_grad()
            
            # Forward pass - get only segmentation output (ignore attention maps)
            pred, _, _, _ = model(ecg)  # d1, x1, x2, x3

            loss = criterion(pred, seg)
            loss.backward()
            optimiser.step()

            train_losses.append(loss.item())


        model.eval()  # prep model for evaluation
        for data in validation_dataloader:
            inp, lab = data
            inp, lab = inp.to(device=device, dtype=torch.float), lab.type(torch.float).to(device)

            inp = inp[:, :2496]  # EXACTLY as original
            inp = inp[:, np.newaxis, :]  # Add channel dimension
            lab = lab[:, :3, :2496].float().to(device)  # EXACTLY as original

            # Forward pass: compute predicted outputs
            output, _, _, _ = model(inp)  # d1, x1, x2, x3
            loss = criterion(output, lab)
            valid_losses.append(loss.item())

        # Calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        scheduler.step(valid_loss)

        epoch_len = len(str(config.TEACHER_CONFIG['max_epochs']))

        print_msg = (f'[{epoch:>{epoch_len}}/{config.TEACHER_CONFIG["max_epochs"]:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')

        print(print_msg)

        # Clear lists to track next epoch
        train_losses = []
        valid_losses = []

        # Early stopping check
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    return model, avg_train_losses, avg_valid_losses


def plot_training_history(train_losses, valid_losses, save_path=None):
    """Plot training history """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(valid_losses) + 1), valid_losses, label='Validation Loss')

    # Find position of lowest validation loss
    minposs = valid_losses.index(min(valid_losses)) + 1
    plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.xlim(0, len(train_losses) + 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


def main():
    """Main training function for teacher network """
    # Load configuration
    config = Config()
    device = config.DEVICE
    
    print(f"Using device: {device}")
    
    # Create model EXACTLY as original AttU_Net(1, 3)
    model = create_teacher_network(config).to(device)
    print(f"Teacher model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Load and preprocess data EXACTLY as original
    print("Loading ECG delineation data...")
    ecg_data, fullmask = load_and_preprocess_teacher_data(config.TEACHER_DATA_PATH)
    
    # Apply augmentation EXACTLY as original
    print("Applying data augmentation...")
    aug_inputs, aug_labels = apply_augmentation_exactly_as_original(ecg_data, fullmask)
    
    # Train-validation split EXACTLY as original
    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(
        aug_inputs, aug_labels,
        random_state=2018,  # EXACTLY as original
        test_size=0.2,      # EXACTLY as original
        shuffle=True
    )
    
    # Convert to tensors
    train_inputs = torch.tensor(train_inputs, dtype=torch.float32)
    validation_inputs = torch.tensor(validation_inputs, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.float32)
    validation_labels = torch.tensor(validation_labels, dtype=torch.float32)
    
    print(f"Training samples: {len(train_inputs)}")
    print(f"Validation samples: {len(validation_inputs)}")
    
    # Create data loaders EXACTLY as original
    batch_size = config.TEACHER_CONFIG['batch_size']  # 64
    
    train_data = TensorDataset(train_inputs, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    validation_data = TensorDataset(validation_inputs, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)
    
    # Train model
    model, train_loss, valid_loss = train_teacher_model_exactly_as_original(
        model, train_dataloader, validation_dataloader, config, device
    )
    
    # Plot training history
    plot_path = os.path.join(config.CHECKPOINT_DIR, 'teacher_training_history.png')
    plot_training_history(train_loss, valid_loss, plot_path)
    
    # Save final model
    final_path = os.path.join(config.CHECKPOINT_DIR, 'teacher_final.pt')
    torch.save(model.state_dict(), final_path)
    print(f"Teacher model training completed. Model saved to {final_path}")


if __name__ == "__main__":
    main()
