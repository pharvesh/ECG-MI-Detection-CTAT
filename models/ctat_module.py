"""
Cross-Task Attention Transfer (CTAT) Module 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CTATModule(nn.Module):
    """
    Cross-Task Attention Transfer Module
    
    This module handles the attention transfer from teacher to student network
    using KL divergence to align attention distributions.
    """
    
    def __init__(self, alpha=0.9, beta=0.1, kl_reduction='batchmean', log_target=True):
        """
        Initialize CTAT module
        
        Args:
            alpha: Weight for classification loss
            beta: Weight for attention transfer loss
            kl_reduction: Reduction method for KL divergence
            log_target: Whether to use log target in KL divergence
        """
        super(CTATModule, self).__init__()
        self.alpha = alpha
        self.beta = beta
        
        # Loss functions
        self.classification_loss = nn.BCEWithLogitsLoss()
        self.attention_loss = nn.KLDivLoss(reduction=kl_reduction, log_target=log_target)
        
    def normalize_attention(self, attention_map):
        """
        Normalize attention maps exactly as in original code
        
        Args:
            attention_map: Raw attention features
            
        Returns:
            Normalized attention maps using F.log_softmax(F.normalize(x, dim=2), 2)
        """
        
        normalized = F.normalize(attention_map, dim=2)
        return F.log_softmax(normalized, dim=2)
    
    def normalize_attention_target(self, attention_map):
        """
        Normalize target attention maps exactly as in original code
        
        Args:
            attention_map: Raw attention features
            
        Returns:
            Normalized attention maps using F.softmax(F.normalize(x, dim=2), 2)
        """
       
        normalized = F.normalize(attention_map, dim=2)
        return F.softmax(normalized, dim=2)
    
    def forward(self, student_output, student_att1, student_att2, student_att3, 
                teacher_att1, teacher_att2, teacher_att3, targets):
        """
        Compute combined loss exactly as in original code
        
        Args:
            student_output: Classification output (student[0])
            student_att1, student_att2, student_att3: Student attention maps (student[1], [2], [3])
            teacher_att1, teacher_att2, teacher_att3: Teacher attention maps (teacher[1], [2], [3])
            targets: Ground truth labels
            
        Returns:
            tuple: (total_loss, classification_loss, attention_loss1, attention_loss2, attention_loss3)
        """
        # Classification loss
        cls_loss = self.classification_loss(student_output, targets.unsqueeze(1))
        
        # Attention transfer losses - exactly as in original
        att_loss1 = self.attention_loss(
            self.normalize_attention(student_att1), 
            self.normalize_attention_target(teacher_att1)
        )
        att_loss2 = self.attention_loss(
            self.normalize_attention(student_att2),
            self.normalize_attention_target(teacher_att2) 
        )
        att_loss3 = self.attention_loss(
            self.normalize_attention(student_att3),
            self.normalize_attention_target(teacher_att3)
        )
        
        # Combined loss exactly as original: 0.9*classification + 0.1*(att1 + att2 + att3)
        total_loss = self.alpha * cls_loss + self.beta * (att_loss1 + att_loss2 + att_loss3)
        
        return total_loss, cls_loss, att_loss1, att_loss2, att_loss3


class EarlyStopping:
    """Early stopping utility class"""
    
    def __init__(self, patience=25, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Initialize early stopping
        
        Args:
            patience: Number of epochs to wait before stopping
            verbose: Whether to print messages
            delta: Minimum change to qualify as improvement
            path: Path to save the best model
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        """
        Check if training should be stopped
        
        Args:
            val_loss: Current validation loss
            model: Model to save if improved
        """
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Save model checkpoint when validation loss improves"""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def create_ctat_module(config):
    """
    Factory function to create CTAT module with configuration
    
    Args:
        config: Configuration object with CTAT parameters
    
    Returns:
        CTATModule instance
    """
    return CTATModule(
        alpha=config.CTAT_CONFIG['alpha'],
        beta=config.CTAT_CONFIG['beta'],
        kl_reduction=config.CTAT_CONFIG['kl_reduction'],
        log_target=config.CTAT_CONFIG['log_target']
    )
