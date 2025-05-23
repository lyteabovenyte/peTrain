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
        
        # Create language encoder
        self.lang_enc = EncoderFactory.create_language_encoder(
            **config['lang_encoder']
        )
        
        # Create visual encoder
        self.vis_enc = EncoderFactory.create_visual_encoder(
            **config['visual_encoder']
        )
        
        # Create cross-modal transformer
        self.attn = CrossModalTransformer(
            visual_dim=config['cross_modal']['visual_dim'],
            lang_dim=config['cross_modal']['lang_dim'],
            hidden_dim=config['cross_modal']['hidden_dim'],
            num_heads=config['cross_modal'].get('num_heads', 8),
            num_layers=config['cross_modal'].get('num_layers', 2),
            dropout=config['cross_modal'].get('dropout', 0.1)
        )
        
        # Create policy network
        self.policy = VLNPolicy(
            input_dim=config['cross_modal']['hidden_dim'],
            hidden_dim=config['policy']['hidden_dim'],
            action_space=config['policy']['action_space']
        )
        
        # Ensure all parameters require gradients
        for param in self.parameters():
            param.requires_grad = True
    
    def forward(self, rgb, depth, instr_tokens):
        """
        rgb: [B,3,H,W], depth: [B,1,H,W], instr_tokens: [B,L]
        Returns: action_logits, state_values
        """
        # Encode language
        lang_feats, _ = self.lang_enc(instr_tokens)  # [B,L,lang_hidden]
        
        # Encode vision
        vis_feats = self.vis_enc(rgb, depth)  # [B, C, H', W']
        
        # Fuse modalities via transformer
        fused = self.attn(vis_feats, lang_feats)  # [B, attn_hidden]
        
        # Policy forward
        logits, value = self.policy(fused)  # [B, action_space], [B,1]
        
        return logits, value.squeeze(1)
    
    @classmethod
    def from_config(cls, config_path):
        """Create agent from configuration file."""
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
        return cls(config['model'])