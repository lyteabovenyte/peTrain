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
    """Lightweight EfficientNet-based visual encoder."""
    def __init__(self, model_name: str = 'efficientnet-b0', **kwargs):
        super().__init__()
        import timm
        # Use smallest EfficientNet variant
        self.model = timm.create_model('efficientnet_b0', pretrained=True)
        # Remove classifier
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        # Add projection to reduce feature dimensions
        self.projection = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.config = {
            'model_name': model_name,
            **kwargs
        }
    
    def forward(self, rgb, depth=None):
        features = self.model(rgb)
        features = self.projection(features)
        return features

@EncoderRegistry.register_visual_encoder('mobilenet')
class MobileNetEncoder(BaseVisualEncoder):
    """Lightweight MobileNet-based visual encoder."""
    def __init__(self, pretrained: bool = True, **kwargs):
        super().__init__()
        # Use MobileNetV2 with new weights parameter
        weights = torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', weights=weights)
        # Remove classifier
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        # Add projection to reduce feature dimensions
        self.projection = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.config = {
            'pretrained': pretrained,
            **kwargs
        }
    
    def forward(self, rgb, depth=None):
        features = self.model(rgb)
        features = self.projection(features)
        return features

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