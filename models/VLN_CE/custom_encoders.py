import torch
import torch.nn as nn
from .encoders import BaseLanguageEncoder, BaseVisualEncoder, EncoderRegistry
import torchvision

@EncoderRegistry.register_language_encoder('gpt2')
class GPT2Encoder(BaseLanguageEncoder):
    """Lightweight GPT-2 based language encoder."""
    def __init__(self, model_name: str = 'gpt2', **kwargs):
        super().__init__()
        from transformers import GPT2Model, GPT2Config
        # Use smaller GPT-2 variant
        config = GPT2Config.from_pretrained(model_name)
        config.n_layer = min(config.n_layer, 3)  # Reduce layers
        config.n_head = min(config.n_head, 4)    # Reduce heads
        self.model = GPT2Model.from_pretrained(model_name, config=config)
        self.hidden_dim = self.model.config.hidden_size
        self.config = {
            'model_name': model_name,
            'hidden_dim': self.hidden_dim,
            **kwargs
        }
    
    def forward(self, tokens):
        with torch.no_grad():  # Disable gradient computation for transformer
            outputs = self.model(tokens)
        return outputs.last_hidden_state, None

@EncoderRegistry.register_visual_encoder('efficientnet')
class EfficientNetEncoder(BaseVisualEncoder):
    """Lightweight EfficientNet-based visual encoder with depth fusion strategies.
    Supports fusion_strategy: 'early', 'mid', 'late', or 'none'.
    """
    def __init__(self, model_name: str = 'efficientnet-b0', use_depth: bool = False, fusion_strategy: str = 'mid', **kwargs):
        super().__init__()
        import timm
        self.use_depth = use_depth
        self.fusion_strategy = fusion_strategy
        # Use smallest EfficientNet variant
        self.rgb_model = timm.create_model('efficientnet_b0', pretrained=True, features_only=True)
        self.rgb_conv = nn.Sequential(
            nn.Conv2d(320, 1280, kernel_size=1),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True)
        )
        self.projection = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.config = {
            'model_name': model_name,
            'use_depth': use_depth,
            'fusion_strategy': fusion_strategy,
            **kwargs
        }
        if use_depth and fusion_strategy == 'early':
            # Early fusion: modify first conv layer to accept 4 channels
            import timm
            model = timm.create_model('efficientnet_b0', pretrained=True)
            conv_stem = model.conv_stem
            new_conv = nn.Conv2d(4, conv_stem.out_channels, kernel_size=conv_stem.kernel_size, stride=conv_stem.stride, padding=conv_stem.padding, bias=False)
            with torch.no_grad():
                new_conv.weight[:, :3] = conv_stem.weight
                new_conv.weight[:, 3:] = conv_stem.weight[:, :1]
            model.conv_stem = new_conv
            self.rgb_model = nn.Sequential(*list(model.children())[:-1])
        if use_depth and fusion_strategy == 'mid':
            self.depth_conv = nn.Sequential(
                nn.Conv2d(1, 320, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(320),
                nn.ReLU(inplace=True)
            )
            self.fusion = nn.Sequential(
                nn.Conv2d(1280 * 2, 256, kernel_size=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )
        if use_depth and fusion_strategy == 'late':
            import timm
            self.depth_model = timm.create_model('efficientnet_b0', pretrained=True, features_only=True)
            self.depth_conv = nn.Sequential(
                nn.Conv2d(320, 1280, kernel_size=1),
                nn.BatchNorm2d(1280),
                nn.ReLU(inplace=True)
            )
            self.late_fusion = nn.Sequential(
                nn.Conv2d(1280 * 2, 1280, kernel_size=1),
                nn.BatchNorm2d(1280),
                nn.ReLU(inplace=True)
            )
    def forward(self, rgb, depth=None):
        if not self.use_depth or self.fusion_strategy == 'none' or depth is None:
            feats = self.rgb_model(rgb)[-1]
            feats = self.rgb_conv(feats)
            feats = self.projection(feats)
            return feats
        if self.fusion_strategy == 'early':
            x = torch.cat([rgb, depth], dim=1)
            feats = self.rgb_model(x)[-1]
            feats = self.rgb_conv(feats)
            feats = self.projection(feats)
            return feats
        elif self.fusion_strategy == 'mid':
            rgb_feats = self.rgb_model(rgb)[-1]
            rgb_feats = self.rgb_conv(rgb_feats)
            depth_feats = self.depth_conv(depth)
            if rgb_feats.size() != depth_feats.size():
                depth_feats = nn.functional.interpolate(depth_feats, size=rgb_feats.shape[2:], mode='bilinear', align_corners=False)
            combined = torch.cat([rgb_feats, depth_feats], dim=1)
            feats = self.fusion(combined)
            return feats
        elif self.fusion_strategy == 'late':
            rgb_feats = self.rgb_model(rgb)[-1]
            rgb_feats = self.rgb_conv(rgb_feats)
            depth_feats = self.depth_model(depth)[-1]
            depth_feats = self.depth_conv(depth_feats)
            if rgb_feats.size() != depth_feats.size():
                depth_feats = nn.functional.interpolate(depth_feats, size=rgb_feats.shape[2:], mode='bilinear', align_corners=False)
            combined = torch.cat([rgb_feats, depth_feats], dim=1)
            feats = self.late_fusion(combined)
            feats = self.projection(feats)
            return feats
        else:
            feats = self.rgb_model(rgb)[-1]
            feats = self.rgb_conv(feats)
            feats = self.projection(feats)
            return feats

@EncoderRegistry.register_visual_encoder('mobilenet')
class MobileNetEncoder(BaseVisualEncoder):
    """Lightweight MobileNet-based visual encoder with depth fusion strategies.
    Supports fusion_strategy: 'early', 'mid', 'late', or 'none'.
    """
    def __init__(self, pretrained: bool = True, use_depth: bool = False, fusion_strategy: str = 'mid', **kwargs):
        super().__init__()
        self.use_depth = use_depth
        self.fusion_strategy = fusion_strategy
        weights = torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        self.rgb_model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', weights=weights)
        self.rgb_model = nn.Sequential(*list(self.rgb_model.children())[:-1])
        self.projection = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.config = {
            'pretrained': pretrained,
            'use_depth': use_depth,
            'fusion_strategy': fusion_strategy,
            **kwargs
        }
        if use_depth and fusion_strategy == 'early':
            import torchvision.models as models
            model = models.mobilenet_v2(weights=weights)
            conv_stem = model.features[0][0]
            new_conv = nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1, bias=False)
            with torch.no_grad():
                new_conv.weight[:, :3] = conv_stem.weight
                new_conv.weight[:, 3:] = conv_stem.weight[:, :1]
            model.features[0][0] = new_conv
            self.rgb_model = nn.Sequential(*list(model.children())[:-1])
        if use_depth and fusion_strategy == 'mid':
            self.depth_conv = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            )
            self.fusion = nn.Sequential(
                nn.Conv2d(1280 + 32, 256, kernel_size=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )
        if use_depth and fusion_strategy == 'late':
            weights = torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
            self.depth_model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', weights=weights)
            self.depth_model = nn.Sequential(*list(self.depth_model.children())[:-1])
            self.late_fusion = nn.Sequential(
                nn.Conv2d(1280 * 2, 1280, kernel_size=1),
                nn.BatchNorm2d(1280),
                nn.ReLU(inplace=True)
            )
    def forward(self, rgb, depth=None):
        if not self.use_depth or self.fusion_strategy == 'none' or depth is None:
            feats = self.rgb_model(rgb)
            feats = self.projection(feats)
            return feats
        if self.fusion_strategy == 'early':
            x = torch.cat([rgb, depth], dim=1)
            feats = self.rgb_model(x)
            feats = self.projection(feats)
            return feats
        elif self.fusion_strategy == 'mid':
            rgb_feats = self.rgb_model(rgb)
            depth_feats = self.depth_conv(depth)
            if rgb_feats.size() != depth_feats.size():
                depth_feats = nn.functional.interpolate(depth_feats, size=rgb_feats.shape[2:], mode='bilinear', align_corners=False)
            combined = torch.cat([rgb_feats, depth_feats], dim=1)
            feats = self.fusion(combined)
            return feats
        elif self.fusion_strategy == 'late':
            rgb_feats = self.rgb_model(rgb)
            depth_feats = self.depth_model(depth)
            if rgb_feats.size() != depth_feats.size():
                depth_feats = nn.functional.interpolate(depth_feats, size=rgb_feats.shape[2:], mode='bilinear', align_corners=False)
            combined = torch.cat([rgb_feats, depth_feats], dim=1)
            feats = self.late_fusion(combined)
            feats = self.projection(feats)
            return feats
        else:
            feats = self.rgb_model(rgb)
            feats = self.projection(feats)
            return feats

# Example of how to use custom encoders:
"""
# In your main script:
from models.VLN_CE.encoders import EncoderFactory

# Load custom encoders
EncoderFactory.load_custom_encoders(['models.VLN_CE.custom_encoders'])

# Create encoders with CPU-friendly settings
lang_encoder = EncoderFactory.create_language_encoder('gpt2')
vis_encoder = EncoderFactory.create_visual_encoder('mobilenet')  # Use MobileNet instead of EfficientNet

# List available encoders
print(EncoderFactory.list_available_encoders())
""" 