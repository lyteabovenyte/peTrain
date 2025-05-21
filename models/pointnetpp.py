"""
    Simplified PointNet++ architecture for point cloud classification.
    Matches the preprocessing.py dataloader and train.py requirements.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Comment out the original implementation
"""
[Previous implementation remains here but commented out]
"""

class SimplePointNetPP(nn.Module):
    def __init__(self, num_classes=123, num_points=1024):
        super(SimplePointNetPP, self).__init__()
        
        # Input: [B, N, 3] -> [B, 3, N]
        self.input_transform = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # First set abstraction layer
        self.sa1 = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
        # Second set abstraction layer
        self.sa2 = nn.Sequential(
            nn.Conv1d(256, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        
        # Global feature layer
        self.global_feat = nn.Sequential(
            nn.Conv1d(1024, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
        # Global pooling
        self.max_pool = lambda x: torch.max(x, 2, keepdim=False)[0]  # [B, C]
        
        # Classification head - adjusted for 123 classes
        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)  # Output: [B, num_classes]
        )
        
    def forward(self, x):
        """
        Input:
            x: point cloud data [B, N, 3]
        Return:
            x: classification scores [B, num_classes]
        """
        # Transpose input to [B, 3, N]
        x = x.transpose(1, 2)
        
        # Initial feature extraction
        x = self.input_transform(x)  # [B, 64, N]
        
        # First set abstraction
        x = self.sa1(x)  # [B, 256, N]
        
        # Second set abstraction
        x = self.sa2(x)  # [B, 1024, N]
        
        # Global feature extraction
        x = self.global_feat(x)  # [B, 256, N]
        
        # Global pooling
        x = self.max_pool(x)  # [B, 256]
        
        # Classification
        x = self.classifier(x)  # [B, num_classes]
        
        # Apply log_softmax
        x = F.log_softmax(x, dim=-1)
        
        return x

# For backward compatibility, make PointNetPP an alias for SimplePointNetPP
PointNetPP = SimplePointNetPP