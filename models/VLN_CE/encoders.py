import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from transformers import AutoModel, AutoConfig
import importlib
import inspect
from typing import Dict, Type, Any, Optional

class EncoderRegistry:
    """Registry for all encoders with dynamic loading capabilities."""
    _language_encoders: Dict[str, Type['BaseLanguageEncoder']] = {}
    _visual_encoders: Dict[str, Type['BaseVisualEncoder']] = {}
    
    @classmethod
    def register_language_encoder(cls, name: str):
        """Decorator to register a language encoder."""
        def decorator(encoder_class: Type['BaseLanguageEncoder']):
            cls._language_encoders[name] = encoder_class
            return encoder_class
        return decorator
    
    @classmethod
    def register_visual_encoder(cls, name: str):
        """Decorator to register a visual encoder."""
        def decorator(encoder_class: Type['BaseVisualEncoder']):
            cls._visual_encoders[name] = encoder_class
            return encoder_class
        return decorator
    
    @classmethod
    def get_language_encoder(cls, name: str) -> Type['BaseLanguageEncoder']:
        """Get a registered language encoder class."""
        if name not in cls._language_encoders:
            raise ValueError(f"Unknown language encoder: {name}")
        return cls._language_encoders[name]
    
    @classmethod
    def get_visual_encoder(cls, name: str) -> Type['BaseVisualEncoder']:
        """Get a registered visual encoder class."""
        if name not in cls._visual_encoders:
            raise ValueError(f"Unknown visual encoder: {name}")
        return cls._visual_encoders[name]
    
    @classmethod
    def list_available_encoders(cls) -> Dict[str, list]:
        """List all available encoders."""
        return {
            'language': list(cls._language_encoders.keys()),
            'visual': list(cls._visual_encoders.keys())
        }
    
    @classmethod
    def load_encoder_module(cls, module_path: str):
        """Dynamically load a module containing encoder implementations."""
        try:
            importlib.import_module(module_path)
        except ImportError as e:
            raise ImportError(f"Failed to load encoder module {module_path}: {e}")

class BaseEncoder(ABC, nn.Module):
    """Base class for all encoders with common functionality."""
    def __init__(self):
        super().__init__()
        self._config = {}
    
    @property
    def config(self) -> dict:
        """Get encoder configuration."""
        return self._config
    
    @config.setter
    def config(self, value: dict):
        """Set encoder configuration."""
        self._config = value
    
    @classmethod
    def from_config(cls, config: dict) -> 'BaseEncoder':
        """Create encoder instance from configuration."""
        return cls(**config)

class BaseLanguageEncoder(BaseEncoder):
    """Abstract base class for language encoders."""
    @abstractmethod
    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, Optional[Any]]:
        """
        Args:
            tokens: Input tokens [B, L]
        Returns:
            features: Encoded features [B, L, hidden_dim]
            state: Optional state information (e.g., LSTM hidden state)
        """
        pass

class BaseVisualEncoder(BaseEncoder):
    """Abstract base class for visual encoders."""
    @abstractmethod
    def forward(self, rgb: torch.Tensor, depth: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            rgb: RGB image [B, 3, H, W]
            depth: Optional depth image [B, 1, H, W]
        Returns:
            features: Encoded visual features [B, C, H', W']
        """
        pass

@EncoderRegistry.register_language_encoder('lstm')
class LSTMEncoder(BaseLanguageEncoder):
    """Lightweight LSTM-based language encoder."""
    def __init__(self, vocab_size: int, embed_dim: int = 128, 
                 hidden_dim: int = 256, num_layers: int = 1, **kwargs):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.config = {
            'vocab_size': vocab_size,
            'embed_dim': embed_dim,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            **kwargs
        }
    
    def forward(self, tokens):
        emb = self.embedding(tokens)
        features, _ = self.lstm(emb)
        return features  # Return only the features, not the tuple

@EncoderRegistry.register_language_encoder('transformer')
class TransformerEncoder(BaseLanguageEncoder):
    """Lightweight transformer-based language encoder."""
    def __init__(self, model_name: str = 'distilbert-base-uncased', 
                 hidden_dim: Optional[int] = None, **kwargs):
        super().__init__()
        # Use smaller models by default
        config = AutoConfig.from_pretrained(model_name)
        config.num_hidden_layers = min(config.num_hidden_layers, 3)  # Reduce layers
        config.num_attention_heads = min(config.num_attention_heads, 4)  # Reduce heads
        self.model = AutoModel.from_pretrained(model_name, config=config)
        
        # Add projection layer to match expected dimensions
        self.hidden_dim = hidden_dim or 256  # Default to 256 if not specified
        self.projection = nn.Linear(self.model.config.hidden_size, self.hidden_dim)
        
        # Initialize projection layer
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        
        self.config = {
            'model_name': model_name,
            'hidden_dim': self.hidden_dim,
            **kwargs
        }
    
    def forward(self, tokens):
        # Create attention mask for padded sequences
        attention_mask = (tokens != 0).long()
        
        # Forward pass with attention mask
        outputs = self.model(tokens, attention_mask=attention_mask)
        
        # Project to expected dimension
        features = self.projection(outputs.last_hidden_state)
        return features

@EncoderRegistry.register_visual_encoder('resnet')
class ResNetEncoder(BaseVisualEncoder):
    """Lightweight ResNet-based visual encoder with depth fusion strategies.
    Supports fusion_strategy: 'early', 'mid', 'late', or 'none'.
    - early: Concatenate depth as 4th channel to RGB and use a modified first conv layer.
    - mid: Process RGB and depth separately, then fuse after initial layers (default, previous behavior).
    - late: Process RGB and depth through separate backbones, then fuse at the feature level.
    - none: Ignore depth.
    """
    def __init__(self, pretrained: bool = False, use_depth: bool = False, fusion_strategy: str = 'mid', **kwargs):
        super().__init__()
        self.use_depth = use_depth
        self.fusion_strategy = fusion_strategy
        # Use smaller ResNet variant
        self.rgb_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=pretrained)
        self.rgb_model = nn.Sequential(*list(self.rgb_model.children())[:-2])  # Remove avgpool and fc
        # Add projection to reduce feature dimensions
        self.projection = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
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
            # Early fusion: modify first conv layer to accept 4 channels
            from torchvision.models.resnet import BasicBlock
            import torchvision.models as models
            backbone = models.resnet18(pretrained=pretrained)
            conv1 = backbone.conv1
            new_conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                new_conv1.weight[:, :3] = conv1.weight
                new_conv1.weight[:, 3:] = conv1.weight[:, :1]  # Copy first channel weights for depth
            self.rgb_model[0] = new_conv1
        if use_depth and fusion_strategy == 'mid':
            # Mid fusion: process depth separately, then fuse
            self.depth_conv = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
            # Initialize depth conv with RGB first-channel weights
            backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=pretrained)
            self.depth_conv[0].weight.data = backbone.conv1.weight.data.mean(dim=1, keepdim=True)
            self.fusion = nn.Sequential(
                nn.Conv2d(512 * 2, 256, kernel_size=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )
        if use_depth and fusion_strategy == 'late':
            # Late fusion: separate backbones for RGB and depth
            self.depth_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=pretrained)
            self.depth_model = nn.Sequential(*list(self.depth_model.children())[:-2])
            self.late_fusion = nn.Sequential(
                nn.Conv2d(512 * 2, 512, kernel_size=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            )
    def forward(self, rgb, depth=None):
        if not self.use_depth or self.fusion_strategy == 'none' or depth is None:
            # Only RGB
            features = self.rgb_model(rgb)
            features = self.projection(features)
            return features
        if self.fusion_strategy == 'early':
            # Early fusion: concatenate depth as 4th channel
            x = torch.cat([rgb, depth], dim=1)  # [B,4,H,W]
            features = self.rgb_model(x)
            features = self.projection(features)
            return features
        elif self.fusion_strategy == 'mid':
            # Mid fusion: process separately, then fuse
            rgb_feats = self.rgb_model(rgb)
            depth_feats = self.depth_conv(depth)
            if rgb_feats.size() != depth_feats.size():
                depth_feats = nn.functional.interpolate(
                    depth_feats, size=rgb_feats.shape[2:], mode='bilinear', align_corners=False)
            combined = torch.cat([rgb_feats, depth_feats], dim=1)
            feats = self.fusion(combined)
            return feats
        elif self.fusion_strategy == 'late':
            # Late fusion: separate backbones, then fuse
            rgb_feats = self.rgb_model(rgb)
            depth_feats = self.depth_model(depth)
            if rgb_feats.size() != depth_feats.size():
                depth_feats = nn.functional.interpolate(
                    depth_feats, size=rgb_feats.shape[2:], mode='bilinear', align_corners=False)
            combined = torch.cat([rgb_feats, depth_feats], dim=1)
            feats = self.late_fusion(combined)
            feats = self.projection(feats)
            return feats
        else:
            # Default to RGB only
            features = self.rgb_model(rgb)
            features = self.projection(features)
            return features

@EncoderRegistry.register_visual_encoder('vit')
class ViTEncoder(BaseVisualEncoder):
    """Lightweight Vision Transformer encoder."""
    def __init__(self, model_name: str = 'google/vit-base-patch16-224', **kwargs):
        super().__init__()
        # Use smaller ViT variant
        config = AutoConfig.from_pretrained(model_name)
        config.num_hidden_layers = min(config.num_hidden_layers, 3)  # Reduce layers
        config.num_attention_heads = min(config.num_attention_heads, 4)  # Reduce heads
        self.model = AutoModel.from_pretrained(model_name, config=config)
        self.config = {
            'model_name': model_name,
            **kwargs
        }
    
    def forward(self, rgb, depth=None):
        with torch.no_grad():  # Disable gradient computation for transformer
            outputs = self.model(rgb)
        return outputs.last_hidden_state

class EncoderFactory:
    """Factory class for creating encoders with dynamic loading support."""
    @staticmethod
    def create_language_encoder(encoder_type: str, **kwargs) -> BaseLanguageEncoder:
        """Create a language encoder instance."""
        encoder_class = EncoderRegistry.get_language_encoder(encoder_type)
        return encoder_class(**kwargs)
    
    @staticmethod
    def create_visual_encoder(encoder_type: str, **kwargs) -> BaseVisualEncoder:
        """Create a visual encoder instance."""
        encoder_class = EncoderRegistry.get_visual_encoder(encoder_type)
        return encoder_class(**kwargs)
    
    @staticmethod
    def load_custom_encoders(module_paths: list[str]):
        """Load custom encoder implementations from specified modules."""
        for path in module_paths:
            EncoderRegistry.load_encoder_module(path)
    
    @staticmethod
    def list_available_encoders() -> dict:
        """List all available encoders."""
        return EncoderRegistry.list_available_encoders() 