import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAttention(nn.Module):
    """
    Cross-modal attention module: attends from language tokens onto visual features.
    """
    def __init__(self, visual_dim, lang_dim, hidden_dim):
        super().__init__()
        # Project both visual and language features to a common dimension
        self.proj_v = nn.Linear(visual_dim, hidden_dim)  # 512 -> 256
        self.proj_l = nn.Linear(lang_dim, hidden_dim)    # 256 -> 256
        # Optionally, an output projection after combining
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Initialize weights for better training stability
        nn.init.xavier_uniform_(self.proj_v.weight)
        nn.init.xavier_uniform_(self.proj_l.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)
    
    def forward(self, visual_feats, lang_feats):
        """
        visual_feats: [B, C_v, H', W'] (spatial visual features)
        lang_feats: [B, L, C_l] (sequence of token features)
        Returns: fused context [B, hidden_dim]
        """
        B, C, H, W = visual_feats.shape
        # Flatten visual features to [B, H'*W', C]
        visual_flat = visual_feats.view(B, C, -1).transpose(1,2)  # [B, N, C]
        
        # Project to common dimension
        V = self.proj_v(visual_flat)  # [B, N, hidden]
        L = self.proj_l(lang_feats)   # [B, L, hidden]
        
        # Compute attention scores: [B, L, N]
        scores = torch.bmm(L, V.transpose(1,2))
        attn_weights = F.softmax(scores, dim=-1)  # along visual tokens
        
        # Attend visual features for each language token: [B, L, hidden]
        attended = torch.bmm(attn_weights, V)
        
        # Pool across tokens (average)
        fused = attended.mean(dim=1)  # [B, hidden]
        fused = F.relu(self.output_proj(fused))
        
        return fused  # combined vision-language feature vector