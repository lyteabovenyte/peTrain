import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

class VisualEncoder(nn.Module):
    """
    Encodes visual observations (RGB+D) into spatial feature maps with proper preprocessing.
    """
    def __init__(self, pretrained=True, use_depth=False, input_size=(224, 224)):
        super().__init__()
        self.input_size = input_size
        self.use_depth = use_depth
        
        # Image normalization
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Use ResNet18 for RGB
        backbone = models.resnet18(pretrained=pretrained)
        layers = list(backbone.children())[:-2]  # remove avgpool and fc
        self.cnn = nn.Sequential(*layers)
        
        if use_depth:
            # Separate depth processing branch
            self.depth_conv = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
            # Initialize depth conv with RGB first-channel weights
            self.depth_conv[0].weight.data = backbone.conv1.weight.data.mean(dim=1, keepdim=True)
            
            # Fusion layer to combine RGB and depth features
            self.fusion = nn.Sequential(
                nn.Conv2d(512 * 2, 512, kernel_size=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            )
    
    def preprocess_rgb(self, rgb):
        """Preprocess RGB images."""
        if not isinstance(rgb, torch.Tensor):
            rgb = torch.from_numpy(rgb).float()
        if rgb.dim() == 3:
            rgb = rgb.unsqueeze(0)
        if rgb.size(1) != 3:
            raise ValueError(f"Expected RGB input with 3 channels, got {rgb.size(1)}")
        # Normalize to [0,1] if needed
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        # Apply ImageNet normalization
        rgb = self.normalize(rgb)
        return rgb
    
    def preprocess_depth(self, depth):
        """Preprocess depth images."""
        if not isinstance(depth, torch.Tensor):
            depth = torch.from_numpy(depth).float()
        if depth.dim() == 3:
            depth = depth.unsqueeze(0)
        if depth.size(1) != 1:
            raise ValueError(f"Expected depth input with 1 channel, got {depth.size(1)}")
        # Normalize depth to [0,1] if needed
        if depth.max() > 1.0:
            depth = depth / depth.max()
        return depth
    
    def forward(self, rgb_img, depth_img=None):
        """
        rgb_img: [B,3,H,W] or numpy array
        depth_img: [B,1,H,W] or numpy array (if use_depth)
        Returns: visual_feats [B, C, H', W']
        """
        # Preprocess RGB
        rgb = self.preprocess_rgb(rgb_img)
        
        if self.use_depth and depth_img is not None:
            # Preprocess depth
            depth = self.preprocess_depth(depth_img)
            
            # Process RGB and depth separately
            rgb_feats = self.cnn(rgb)
            depth_feats = self.depth_conv(depth)
            
            # Ensure feature maps have same spatial dimensions
            if rgb_feats.size() != depth_feats.size():
                depth_feats = nn.functional.interpolate(
                    depth_feats, 
                    size=rgb_feats.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
            
            # Concatenate and fuse features
            combined = torch.cat([rgb_feats, depth_feats], dim=1)
            feats = self.fusion(combined)
        else:
            # Process RGB only
            feats = self.cnn(rgb)
        
        return feats  # [B, 512, H', W']