"""
Training script for Student Network with Cross-Task Attention Transfer (CTAT)
"""
import torch
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from models.teacher_network import create_teacher_network
from models.student_network import create_student_network
from models.ctat_module import create_ctat_module, EarlyStopping
from utils.data_utils import ECGDataset, load_ptbxl_data, prepare_mi_classification_data


def create_student_dataloaders(train_inputs, train_labels, val_inputs, val_labels, data_path, batch_size=32):
    """Create data loaders for student training"""
    
    # Training data loader
    train_dataset = ECGDataset(train_inputs, train_labels, data_path + "train/")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Validation data loader
    val_dataset = ECGDataset(val_inputs, val_labels, data_path + "train/")  # Same path structure
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


def train_student_with_ctat(teacher_model, student_model, ctat_module, train_loader, val_loader, config, device):
    """Train student model with CTAT framework """
    
    # Setup optimizer and scheduler - EXACTLY as original
    optimizer = SGD(
        student_model.parameters(),
        lr=config.STUDENT_CONFIG['learning_rate'],
        momentum=config.STUDENT_CONFIG['momentum'],
        weight_decay=config.STUDENT_CONFIG['weight_decay']
    )
    
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=True)
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.STUDENT_CONFIG['patience'],
        verbose=True,
        path=os.path.join(config.CHECKPOINT_DIR, 'student_best.pt')
    )
    
    # Freeze teacher model - EXACTLY as original
    for param in teacher_model.parameters():
        param.requires_grad = False
    teacher_model.eval()
    
    # Training history
    train_losses = []
    val_losses = []
    train_cls_losses = []
    train_att_losses = []
    
    print("Starting student network training with CTAT...")
    
    for epoch in trange(1, config.STUDENT_CONFIG['max_epochs'] + 1):
        # Training phase
        student_model.train()
        epoch_train_losses = []
        epoch_cls_losses = []
        epoch_att_losses = []
        
        for batch_idx, (ecg_data, labels) in enumerate(train_loader):
            ecg_data = ecg_data.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)
            
            optimizer.zero_grad()
            
            # Forward pass through student - EXACTLY as original
            outputstu = student_model(ecg_data[:, :, :2496])  # Returns 4 values: out1, att2, att3, att4
            
            # Forward pass through teacher - EXACTLY as original  
            lead1seg = ecg_data[:, 1, :2496]  # Use lead II as in original
            outputtea = teacher_model(lead1seg[:, np.newaxis, :])  # Returns d1, x1, x2, x3
            
            # Compute losses - EXACTLY as original
            total_loss, cls_loss, att_loss1, att_loss2, att_loss3 = ctat_module(
                outputstu[0],  # student classification output
                outputstu[1], outputstu[2], outputstu[3],  # student attention maps
                outputtea[1], outputtea[2], outputtea[3],  # teacher attention maps
                labels
            )
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Record losses
            epoch_train_losses.append(total_loss.item())
            epoch_cls_losses.append(cls_loss.item())
            epoch_att_losses.append((att_loss1.item() + att_loss2.item() + att_loss3.item()) / 3)
        
        # Validation phase - EXACTLY as original
        student_model.eval()
        epoch_val_losses = []
        
        with torch.no_grad():
            for batch_idx, (ecg_data, labels) in enumerate(val_loader):
                ecg_data = ecg_data.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.float)
                
                # Forward passes
                outputstu = student_model(ecg_data[:, :, :2496])
                lead1seg = ecg_data[:, 1, :2496]
                outputtea = teacher_model(lead1seg[:, np.newaxis, :])
                
                # Compute loss
                total_loss, _, _, _, _ = ctat_module(
                    outputstu[0], outputstu[1], outputstu[2], outputstu[3],
                    outputtea[1], outputtea[2], outputtea[3], labels
                )
                
                epoch_val_losses.append(total_loss.item())
        
        # Calculate average losses
        train_loss = np.mean(epoch_train_losses)
        val_loss = np.mean(epoch_val_losses)
        cls_loss_avg = np.mean(epoch_cls_losses)
        att_loss_avg = np.mean(epoch_att_losses)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_cls_losses.append(cls_loss_avg)
        train_att_losses.append(att_loss_avg)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Print progress - EXACTLY as original format
        if epoch % 2 == 0:
            print(f'\n\n{cls_loss_avg:.6f} : {att_loss_avg:.6f}')
        
        print(f'Epoch [{epoch}/{config.STUDENT_CONFIG["max_epochs"]}] '
              f'train_loss: {train_loss:.5f} valid_loss: {val_loss:.5f}')
        
        # Early stopping check
        early_stopping(val_loss, student_model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    training_history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_cls_losses': train_cls_losses,
        'train_att_losses': train_att_losses
    }
    
    return student_model, training_history


def plot_ctat_training_history(training_history, save_path=None):
    """Plot CTAT training history with multiple loss components"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Total loss
    axes[0, 0].plot(training_history['train_losses'], label='Training')
    axes[0, 0].plot(training_history['val_losses'], label='Validation')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Classification loss
    axes[0, 1].plot(training_history['train_cls_losses'], label='Training')
    axes[0, 1].plot(training_history['val_cls_losses'], label='Validation')
    axes[0, 1].set_title('Classification Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Attention loss
    axes[1, 0].plot(training_history['train_att_losses'], label='Training')
    axes[1, 0].plot(training_history['val_att_losses'], label='Validation')
    axes[1, 0].set_title('Attention Transfer Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Loss components comparison
    axes[1, 1].plot(training_history['train_cls_losses'], label='Classification Loss')
    axes[1, 1].plot(training_history['train_att_losses'], label='Attention Loss')
    axes[1, 1].set_title('Training Loss Components')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def load_pretrained_teacher(config, device):
    """Load pre-trained teacher model"""
    teacher_model = create_teacher_network(config).to(device)
    
    if os.path.exists(config.TEACHER_MODEL_PATH):
        print(f"Loading pre-trained teacher model from {config.TEACHER_MODEL_PATH}")
        teacher_model.load_state_dict(torch.load(config.TEACHER_MODEL_PATH, map_location=device))
    else:
        print("Warning: Pre-trained teacher model not found. Please train teacher model first.")
        return None
    
    # Freeze teacher model
    for param in teacher_model.parameters():
        param.requires_grad = False
    teacher_model.eval()
    
    return teacher_model


def main():
    """Main training function for student network with CTAT"""
    # Load configuration
    config = Config()
    device = config.DEVICE
    
    print(f"Using device: {device}")
    
    # Load pre-trained teacher model
    teacher_model = load_pretrained_teacher(config, device)
    if teacher_model is None:
        return
    
    # Create student model
    student_model = create_student_network(config).to(device)
    print(f"Student model created with {sum(p.numel() for p in student_model.parameters())} parameters")
    
    # Create CTAT module
    ctat_module = create_ctat_module(config).to(device)
    
    # Load PTB-XL dataset
    print("Loading PTB-XL dataset...")
    Y = load_ptbxl_data(config.STUDENT_DATA_PATH)
    
    # Prepare MI classification data
    train_inputs, train_labels, val_inputs, val_labels = prepare_mi_classification_data(
        Y, config.STUDENT_DATA_PATH,
        test_fold=config.DATA_CONFIG['test_fold'],
        validation_fold=config.DATA_CONFIG['validation_fold']
    )
    
    print(f"Training samples: {len(train_inputs)} (MI: {sum(train_labels)}, Normal: {len(train_labels) - sum(train_labels)})")
    print(f"Validation samples: {len(val_inputs)} (MI: {sum(val_labels)}, Normal: {len(val_labels) - sum(val_labels)})")
    
    # Create data loaders
    train_loader, val_loader = create_student_dataloaders(
        train_inputs, train_labels, val_inputs, val_labels,
        config.STUDENT_DATA_PATH,
        batch_size=config.STUDENT_CONFIG['batch_size']
    )
    
    # Train student model with CTAT
    student_model, training_history = train_student_with_ctat(
        teacher_model, student_model, ctat_module, train_loader, val_loader, config, device
    )
    
    # Plot training history
    plot_path = os.path.join(config.CHECKPOINT_DIR, 'student_ctat_training_history.png')
    plot_ctat_training_history(training_history, plot_path)
    
    # Save final model
    final_path = os.path.join(config.CHECKPOINT_DIR, 'student_final.pt')
    torch.save(student_model.state_dict(), final_path)
    print(f"Student model training completed. Model saved to {final_path}")
    
    # Save training history
    history_path = os.path.join(config.CHECKPOINT_DIR, 'training_history.npz')
    np.savez(history_path, **training_history)
    print(f"Training history saved to {history_path}")


if __name__ == "__main__":
    main()
