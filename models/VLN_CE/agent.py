import torch
import torch.nn as nn
from .encoders import EncoderFactory
from .cross_modal import CrossModalTransformer
from .policy import VLNPolicy

class VLNCEAgent(nn.Module):
    """
    Encapsulates the VLN-CE agent with plug-and-play components.
    """
    def __init__(self, config):
        """
        Args:
            config: Dictionary containing model configuration
                Required keys:
                - lang_encoder: dict with 'type' and other parameters
                - visual_encoder: dict with 'type' and other parameters
                - cross_modal: dict with transformer parameters
                - policy: dict with policy parameters
        """
        super().__init__()
        self.config = config
        
        # Initialize encoders
        self.visual_encoder = EncoderFactory.create_visual_encoder(
            config['visual_encoder']['type'],
            **config['visual_encoder']
        )
        
        self.language_encoder = EncoderFactory.create_language_encoder(
            config['lang_encoder']['type'],
            **config['lang_encoder']
        )
        
        # Get encoder output dimensions
        self.visual_dim = config['cross_modal']['visual_dim']
        self.lang_dim = config['cross_modal']['lang_dim']
        self.hidden_dim = config['cross_modal']['hidden_dim']
        
        # Cross-modal transformer
        self.cross_modal = CrossModalTransformer(
            visual_dim=self.visual_dim,
            lang_dim=self.lang_dim,
            hidden_dim=self.hidden_dim,
            num_heads=config['cross_modal']['num_heads'],
            num_layers=config['cross_modal']['num_layers'],
            dropout=config['cross_modal']['dropout']
        )
        
        # Policy network
        self.policy = VLNPolicy(
            input_dim=self.hidden_dim,
            hidden_dim=config['policy']['hidden_dim'],
            action_space=config['policy']['action_space']
        )
        
        # Ensure all parameters require gradients
        for param in self.parameters():
            param.requires_grad = True
    
    def forward(self, rgb, depth, instr):
        """
        rgb: [B, 3, H, W]
        depth: [B, 1, H, W]
        instr: [B, L]
        """
        # Encode visual features
        visual_feats = self.visual_encoder(rgb, depth)  # [B, C_v, H', W']
        
        # Encode language features
        lang_feats = self.language_encoder(instr)  # [B, L, C_l]
        
        # Cross-modal fusion
        fused = self.cross_modal(visual_feats, lang_feats)  # [B, hidden_dim]
        
        # Policy and value predictions
        logits, value = self.policy(fused)  # [B, action_space], [B, 1]
        
        return logits, value
    
    @classmethod
    def from_config(cls, config_path):
        """Create agent from configuration file."""
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
        return cls(config['model'])