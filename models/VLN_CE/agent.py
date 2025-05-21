import torch
import torch.nn as nn
from .lang_encoder import LanguageEncoder
from .visual_encoder import VisualEncoder
from .cross_modal import CrossModalAttention
from .policy import VLNPolicy

class VLNCEAgent(nn.Module):
    """
    Encapsulates the VLN-CE agent: visual encoder, language encoder, cross-modal attention, and policy.
    """
    def __init__(self, vocab_size, embed_dim=300, lang_hidden=512,
                 visual_dim=512, attn_hidden=512, policy_hidden=256, action_space=6):
        super().__init__()
        self.lang_enc = LanguageEncoder(vocab_size, embed_dim, lang_hidden)
        self.vis_enc = VisualEncoder(pretrained=False, use_depth=False)
        self.attn = CrossModalAttention(visual_dim, lang_hidden, attn_hidden)
        self.policy = VLNPolicy(attn_hidden, policy_hidden, action_space)
        
        # Ensure all parameters require gradients
        for param in self.parameters():
            param.requires_grad = True
    
    def forward(self, rgb, depth, instr_tokens):
        """
        rgb: [B,3,H,W], depth: [B,1,H,W], instr_tokens: [B,L]
        Returns: action_logits, state_values
        """
        # Encode language
        lang_feats, (h, c) = self.lang_enc(instr_tokens)  # [B,L,lang_hidden]
        # Encode vision
        vis_feats = self.vis_enc(rgb, depth)             # [B, C, H', W']
        # Fuse modalities via attention
        fused = self.attn(vis_feats, lang_feats)         # [B, attn_hidden]
        # Policy forward
        logits, value = self.policy(fused)               # [B, action_space], [B,1]
        return logits, value.squeeze(1)