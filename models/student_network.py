"""
Student Network: ResNet with Cross-Task Attention Transfer for MI Classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BasicBlock(nn.Module):
    """Basic ResNet block"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    """Bottleneck ResNet block"""
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class StudentAttentionBlock(nn.Module):
    """Attention block for student network that matches teacher attention structure"""
    
    def __init__(self, F_g, F_l, F_int):
        super(StudentAttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv1d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv1d(F_l, F_int, kernel_size=1, stride=2, padding=0, bias=True),
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
        Forward pass through student attention block
        
        Args:
            g: Gating signal (current stage output)
            x: Input features (previous stage output)
        
        Returns:
            Attention-weighted features
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        # Upsample attention to match input size
        resample = nn.Upsample(scale_factor=2)
        psi = resample(psi)
        
        return x * psi


class ConvBlock(nn.Module):
    """Convolutional block for upsampling in attention transfer"""
    
    def __init__(self, ch_in, ch_out):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(ch_in, ch_out, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm1d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class StudentNetwork(nn.Module):
    """
    Student Network: ResNet34 with Cross-Task Attention Transfer for MI Classification
    
    This network learns to classify MI from 12-lead ECG signals while incorporating
    morphological knowledge from the teacher network through attention transfer.
    """
    
    def __init__(self, block, num_blocks, input_channels=12, num_classes=1, dropout_rate=0.2):
        super(StudentNetwork, self).__init__()
        self.in_planes = 64

        # Initial convolution
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        
        # ResNet layers
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        # Attention blocks for cross-task attention transfer
        self.att2 = StudentAttentionBlock(F_g=128, F_l=64, F_int=64)
        self.att3 = StudentAttentionBlock(F_g=256, F_l=128, F_int=128)
        self.att4 = StudentAttentionBlock(F_g=512, F_l=256, F_int=256)
        
        # Upsampling blocks for attention transfer
        self.up2 = ConvBlock(ch_in=64, ch_out=128)
        self.up3 = ConvBlock(ch_in=128, ch_out=256)
        self.up4 = ConvBlock(ch_in=256, ch_out=512)
        
        # Final layers
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
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

    def _make_layer(self, block, planes, num_blocks, stride):
        """Create ResNet layer"""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through student network
        
        Args:
            x: Input 12-lead ECG signal [batch_size, 12, sequence_length]
        
        Returns:
            tuple: (out1, att2, att3, att4) - MATCHES ORIGINAL CODE EXACTLY
                - out1: MI classification logits [batch_size, 1]
                - att2, att3, att4: Attention features for knowledge distillation
        """
        # Initial convolution
        out = F.relu(self.bn1(self.conv1(x)))
        
        # Layer 1
        x1 = self.layer1(out)
        
        # Layer 2 with attention
        x2 = self.layer2(x1)
        att2 = self.att2(g=x2, x=x1)
        att2_up = self.up2(att2)
        x2 = x2 + att2_up
        
        # Layer 3 with attention
        x3 = self.layer3(x2)
        att3 = self.att3(g=x3, x=x2)
        att3_up = self.up3(att3)
        x3 = x3 + att3_up
        
        # Layer 4 with attention
        x4 = self.layer4(x3)
        att4 = self.att4(g=x4, x=x3)
        att4_up = self.up4(att4)
        x4 = x4 + att4_up
        
        # Global average pooling and classification
        out1 = self.avgpool(x4)
        out1 = out1.view(out1.size(0), -1)
        out1 = self.linear(out1)
        out1 = self.dropout(out1)
        
        # Return exactly as in original code: out1, att2, att3, att4
        return out1, att2, att3, att4

    def predict(self, x):
        """
        Make predictions without returning attention maps
        
        Args:
            x: Input 12-lead ECG signal
        
        Returns:
            MI classification probabilities
        """
        out, _ = self.forward(x)
        return torch.sigmoid(out)





def create_resnet34(input_channels=12, num_classes=1, dropout_rate=0.2):
    """Create ResNet34 student network"""
    return StudentNetwork(BasicBlock, [3, 4, 6, 3], input_channels, num_classes, dropout_rate)




def create_student_network(config):
    """
    
    Args:
        config: Configuration object with model parameters
    
    Returns:
        StudentNetwork instance
    """
    return create_resnet34(
        input_channels=config.STUDENT_CONFIG['input_channels'],
        num_classes=config.STUDENT_CONFIG['num_classes'],
        dropout_rate=config.STUDENT_CONFIG['dropout_rate']
    )
