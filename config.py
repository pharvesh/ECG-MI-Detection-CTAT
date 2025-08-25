"""
Configuration file for ECG MI Detection with Cross-Task Attention Transfer (CTAT)
"""
import torch

class Config:
    """Configuration class for hyperparameters and paths"""
    
    # Device configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data paths - UPDATE THESE TO MATCH YOUR SYSTEM
    TEACHER_DATA_PATH = #
    STUDENT_DATA_PATH = #
    CHECKPOINT_DIR = #
    
    # Model paths - UPDATE THESE TO MATCH YOUR SYSTEM  
    TEACHER_MODEL_PATH = #
    STUDENT_MODEL_PATH = #
    
    # Teacher network hyperparameters 
    TEACHER_CONFIG = {
        'input_channels': 1,
        'output_channels': 3,
        'batch_size': 64,  
        'learning_rate': 0.001,  
        'momentum': 0.9,  
        'weight_decay': 1e-5,  
        'dropout_rate': 0.1,  
        'max_epochs': 400,  
        'patience': 40  
    }
    
    # Student network hyperparameters 
    STUDENT_CONFIG = {
        'input_channels': 12,
        'num_classes': 1,
        'batch_size': 32,  
        'learning_rate': 0.01, 
        'momentum': 0.9,  
        'weight_decay': 5e-3,  
        'dropout_rate': 0.2,  
        'max_epochs': 300,  
        'patience': 40  
    }
    
    # CTAT hyperparameters 
    CTAT_CONFIG = {
        'alpha': 0.9,  
        'beta': 0.1,   
        'kl_reduction': 'batchmean',  
        'log_target': True  
    }
    
    # Data preprocessing 
    DATA_CONFIG = {
        'sampling_rate': 250,  
        'segment_length': 2500,  
        'target_length': 2496,  
        'test_fold': 10,  
        'validation_fold': 9  
    }
    
    # Augmentation parameters 
    AUGMENTATION_CONFIG = {
        'gaussian_min_snr': 0.01,  
        'gaussian_max_snr': 0.1,  
        'spike_snr': 25,  
        'spike_period': 250,  
        'highpass_cutoff': 0.5  
    }
    
    # Teacher network architecture 
    TEACHER_ARCH = {
        'encoder_channels': [64, 128, 256, 512, 1024],
        'attention_channels': [256, 128, 64, 32],  
        'maxpool_kernel': 2,  
        'conv_kernel': 3,  
        'padding': 1  
    }
    
    # Student network architecture 
    STUDENT_ARCH = {
        'resnet_blocks': [3, 4, 6, 3],  
        'attention_F_int': [64, 128, 256],  
        'initial_channels': 64,  
        'stage_channels': [64, 128, 256, 512] 
    }
