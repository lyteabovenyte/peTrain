import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        
        # Linear projections and reshape for multi-head attention
        k = self.k_linear(k).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)
        q = self.q_linear(q).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        
        return self.out(out)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class CrossModalTransformer(nn.Module):
    """
    Cross-modal transformer module inspired by ViLBERT architecture.
    Processes visual and language features through transformer layers with cross-attention.
    """
    def __init__(self, visual_dim, lang_dim, hidden_dim, num_heads=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Projection layers for visual and language features
        self.vis_proj = nn.Linear(visual_dim, hidden_dim)
        self.lang_proj = nn.Linear(lang_dim, hidden_dim)
        
        # Position embeddings for visual features
        self.vis_pos_embed = nn.Parameter(torch.randn(1, 100, hidden_dim))  # Max 100 visual tokens
        
        # Cross-modal transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, visual_feats, lang_feats):
        """
        visual_feats: [B, C_v, H', W'] (spatial visual features)
        lang_feats: [B, L, C_l] (sequence of token features)
        Returns: fused context [B, hidden_dim]
        """
        B, C, H, W = visual_feats.shape
        
        # Flatten and project visual features
        visual_flat = visual_feats.view(B, C, -1).transpose(1, 2)  # [B, H'*W', C]
        visual_proj = self.vis_proj(visual_flat)  # [B, H'*W', hidden_dim]
        
        # Add position embeddings to visual features
        visual_proj = visual_proj + self.vis_pos_embed[:, :visual_proj.size(1), :]
        
        # Project language features
        lang_proj = self.lang_proj(lang_feats)  # [B, L, hidden_dim]
        
        # Concatenate features for transformer processing
        combined = torch.cat([visual_proj, lang_proj], dim=1)  # [B, H'*W' + L, hidden_dim]
        
        # Process through transformer layers
        for layer in self.transformer_layers:
            combined = layer(combined)
        
        # Pool the combined features (using mean pooling)
        fused = combined.mean(dim=1)  # [B, hidden_dim]
        
        # Final projection
        fused = F.relu(self.output_proj(fused))
        
        return fused  # [B, hidden_dim]