"""
Configuration file for ECG MI Detection with Cross-Task Attention Transfer (CTAT)
EXACTLY matching original hyperparameters
"""
import torch

class Config:
    """Configuration class for hyperparameters and paths"""
    
    # Device configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data paths - UPDATE THESE TO MATCH YOUR SYSTEM
    TEACHER_DATA_PATH = '/workspace/dandapat/salman/diagnostic/seg1/data/'
    STUDENT_DATA_PATH = '/workspace/dandapat/salman/diagnostic/PTB XL/'
    CHECKPOINT_DIR = '/workspace/dandapat/salman/diagnostic/checkpoint/'
    
    # Model paths - UPDATE THESE TO MATCH YOUR SYSTEM  
    TEACHER_MODEL_PATH = '/workspace/dandapat/salman/diagnostic/saved/seg_new.pt'
    STUDENT_MODEL_PATH = '/workspace/dandapat/salman/diagnostic/checkpointpart2/AtTf_SGD3_0.7_0.3.pt'
    
    # Teacher network hyperparameters - EXACTLY as original
    TEACHER_CONFIG = {
        'input_channels': 1,
        'output_channels': 3,
        'batch_size': 64,  # EXACTLY as original
        'learning_rate': 0.001,  # EXACTLY as original SGD lr=0.001
        'momentum': 0.9,  # EXACTLY as original
        'weight_decay': 1e-5,  # EXACTLY as original weight_decay=0.00001
        'dropout_rate': 0.1,  # EXACTLY as original dropout=0.1
        'max_epochs': 400,  # EXACTLY as original
        'patience': 40  # EXACTLY as original
    }
    
    # Student network hyperparameters - EXACTLY as original
    STUDENT_CONFIG = {
        'input_channels': 12,
        'num_classes': 1,
        'batch_size': 32,  # EXACTLY as original loader batch_size=32
        'learning_rate': 0.01,  # EXACTLY as original SGD lr=0.01
        'momentum': 0.9,  # EXACTLY as original
        'weight_decay': 5e-3,  # EXACTLY as original weight_decay=5e-3
        'dropout_rate': 0.2,  # EXACTLY as original dropout=0.2
        'max_epochs': 300,  # EXACTLY as original
        'patience': 40  # EXACTLY as original
    }
    
    # CTAT hyperparameters - EXACTLY as original
    CTAT_CONFIG = {
        'alpha': 0.9,  # EXACTLY as original: 0.9*lossmain
        'beta': 0.1,   # EXACTLY as original: 0.1*(lossatt1 + lossatt2 + lossatt3)
        'kl_reduction': 'batchmean',  # EXACTLY as original KLDivLoss
        'log_target': True  # EXACTLY as original log_target=True
    }
    
    # Data preprocessing - EXACTLY as original
    DATA_CONFIG = {
        'sampling_rate': 250,  # EXACTLY as original downsampled to 250Hz
        'segment_length': 2500,  # EXACTLY as original signal.resample(ecgdata, 2500)
        'target_length': 2496,  # EXACTLY as original [:, 0:2496]
        'test_fold': 10,  # EXACTLY as original
        'validation_fold': 9  # EXACTLY as original
    }
    
    # Augmentation parameters - EXACTLY as original
    AUGMENTATION_CONFIG = {
        'gaussian_min_snr': 0.01,  # EXACTLY as original gaussian(x, 0.01, 0.1)
        'gaussian_max_snr': 0.1,   # EXACTLY as original
        'spike_snr': 25,  # EXACTLY as original random_spikes(x, 25, 250)
        'spike_period': 250,  # EXACTLY as original
        'highpass_cutoff': 0.5  # EXACTLY as original butter_highpass_filter(x, 0.5, 250)
    }
    
    # Teacher network architecture - EXACTLY as original AttU_Net
    TEACHER_ARCH = {
        'encoder_channels': [64, 128, 256, 512, 1024],  # EXACTLY as original Conv1-Conv5
        'attention_channels': [256, 128, 64, 32],  # EXACTLY as original Att5-Att2 F_int
        'maxpool_kernel': 2,  # EXACTLY as original MaxPool1d(kernel_size=2, stride=2)
        'conv_kernel': 3,  # EXACTLY as original Conv1d kernel_size=3
        'padding': 1  # EXACTLY as original padding=1
    }
    
    # Student network architecture - EXACTLY as original ResNet34
    STUDENT_ARCH = {
        'resnet_blocks': [3, 4, 6, 3],  # EXACTLY as original ResNet34 blocks
        'attention_F_int': [64, 128, 256],  # EXACTLY as original Att2, Att3, Att4
        'initial_channels': 64,  # EXACTLY as original conv1 output
        'stage_channels': [64, 128, 256, 512]  # EXACTLY as original layer1-4
    }