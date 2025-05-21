import torch.nn as nn

class LanguageEncoder(nn.Module):
    """
    Encodes instruction tokens with an embedding layer and an LSTM.
    """
    def __init__(self, vocab_size, embed_dim=300, hidden_dim=512, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # The LSTM’s output sequence seq_feats will be used for cross-modal attention.
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
    
    def forward(self, instr_tokens):
        """
        instr_tokens: [B, L] LongTensor of token indices.
        Returns: seq_feats [B, L, hidden_dim], final_state (h, c).
        """
        emb = self.embedding(instr_tokens)      # [B, L, embed_dim]
        seq_feats, (h, c) = self.lstm(emb)     # [B, L, hidden_dim]
        return seq_feats, (h, c)  # we keep the sequence for attention, and final state if needed