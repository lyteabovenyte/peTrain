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
        features, (h, c) = self.lstm(emb)
        return features, (h, c)

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
        self.hidden_dim = hidden_dim or self.model.config.hidden_size
        self.config = {
            'model_name': model_name,
            'hidden_dim': self.hidden_dim,
            **kwargs
        }
    
    def forward(self, tokens):
        with torch.no_grad():  # Disable gradient computation for transformer
            outputs = self.model(tokens)
        return outputs.last_hidden_state, None

@EncoderRegistry.register_visual_encoder('resnet')
class ResNetEncoder(BaseVisualEncoder):
    """Lightweight ResNet-based visual encoder."""
    def __init__(self, pretrained: bool = False, use_depth: bool = False, **kwargs):
        super().__init__()
        self.use_depth = use_depth
        # Use smaller ResNet variant
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=pretrained)
        # Remove unnecessary layers
        self.model = nn.Sequential(*list(self.model.children())[:-2])  # Remove avgpool and fc
        # Add projection to reduce feature dimensions
        self.projection = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.config = {
            'pretrained': pretrained,
            'use_depth': use_depth,
            **kwargs
        }
    
    def forward(self, rgb, depth=None):
        # Process RGB
        features = self.model(rgb)
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