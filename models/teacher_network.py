"""
Teacher Network: Attention U-Net for ECG Delineation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConvBlock(nn.Module):
    """Convolutional block with residual connection"""
    
    def __init__(self, ch_in, ch_out):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv1d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(ch_out),
            nn.ReLU(inplace=True)
        )
        self.shortcut = nn.Conv1d(ch_in, ch_out, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        x1 = self.conv(x)
        s = self.shortcut(x)
        skip = x1 + s
        return skip


class UpConv(nn.Module):
    """Upsampling convolutional block"""
    
    def __init__(self, ch_in, ch_out):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv1d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)


class AttentionGate(nn.Module):
    """Attention gate mechanism"""
    
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv1d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv1d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv1d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        Forward pass through attention gate
        
        Args:
            g: Gating signal from decoder
            x: Feature maps from encoder
        
        Returns:
            Attention-weighted feature maps
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class TeacherNetwork(nn.Module):
    """
    Teacher Network: Attention U-Net for ECG Delineation
    
    This network learns to segment ECG waveforms into P-waves, QRS complexes, and T-waves.
    The attention mechanisms learned here will be transferred to the student network.
    """
    
    def __init__(self, input_ch=1, output_ch=3, dropout_rate=0.1):
        super(TeacherNetwork, self).__init__()
        
        # Encoder path
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv1 = ConvBlock(ch_in=input_ch, ch_out=64)
        self.conv2 = ConvBlock(ch_in=64, ch_out=128)
        self.conv3 = ConvBlock(ch_in=128, ch_out=256)
        self.conv4 = ConvBlock(ch_in=256, ch_out=512)
        self.conv5 = ConvBlock(ch_in=512, ch_out=1024)
        
        # Decoder path with attention
        self.up5 = UpConv(ch_in=1024, ch_out=512)
        self.att5 = AttentionGate(F_g=512, F_l=512, F_int=256)
        self.up_conv5 = ConvBlock(ch_in=1024, ch_out=512)
        
        self.up4 = UpConv(ch_in=512, ch_out=256)
        self.att4 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.up_conv4 = ConvBlock(ch_in=512, ch_out=256)
        
        self.up3 = UpConv(ch_in=256, ch_out=128)
        self.att3 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.up_conv3 = ConvBlock(ch_in=256, ch_out=128)
        
        self.up2 = UpConv(ch_in=128, ch_out=64)
        self.att2 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.up_conv2 = ConvBlock(ch_in=128, ch_out=64)
        
        # Output layer
        self.conv_1x1 = nn.Conv1d(64, output_ch, kernel_size=1, stride=1, padding=0)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through teacher network
        
        Args:
            x: Input ECG signal [batch_size, 1, sequence_length]
        
        Returns:
            tuple: (d1, x1, x2, x3) - MATCHES ORIGINAL CODE EXACTLY
                - d1: Segmentation output [batch_size, 3, sequence_length] 
                - x1, x2, x3: Intermediate features for attention transfer
        """
        # Encoder path
        x1 = self.conv1(x)
        
        x2 = self.maxpool(x1)
        x2 = self.conv2(x2)
        
        x3 = self.maxpool(x2)
        x3 = self.conv3(x3)
        
        x4 = self.maxpool(x3)
        x4 = self.conv4(x4)
        
        x5 = self.maxpool(x4)
        x5 = self.conv5(x5)
        x5 = self.dropout(x5)
        
        # Decoder path with attention
        d5 = self.up5(x5)
        x4 = self.att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.up_conv5(d5)
        
        d4 = self.up4(d5)
        x3 = self.att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.up_conv4(d4)
        
        d3 = self.up3(d4)
        x2 = self.att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.up_conv3(d3)
        
        d2 = self.up2(d3)
        x1 = self.att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.up_conv2(d2)
        
        # Output
        d1 = self.conv_1x1(d2)
        
        # Return exactly as in original code: d1, x1, x2, x3
        return d1, x1, x2, x3
    
    def get_attention_maps(self, x):
        """
        Extract attention maps without computing gradients
        Used during student network training
        
        Args:
            x: Input ECG signal
        
        Returns:
            List of attention maps
        """
        with torch.no_grad():
            _, attention_maps = self.forward(x)
        return attention_maps


def create_teacher_network(config):
    """
    
    
    Args:
        config: Configuration object with model parameters
    
    Returns:
        TeacherNetwork instance
    """
    return TeacherNetwork(
        input_ch=config.TEACHER_CONFIG['input_channels'],
        output_ch=config.TEACHER_CONFIG['output_channels'],
        dropout_rate=config.TEACHER_CONFIG['dropout_rate']
    )
