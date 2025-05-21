import torch.nn as nn

class VLNPolicy(nn.Module):
    """
    Policy network producing action logits and value estimates from fused features.
    """
    def __init__(self, input_dim, hidden_dim, action_space=6):
        """
        input_dim: dimensionality of fused vision-language feature (from CrossModalAttention)
        hidden_dim: hidden size for the policy MLP
        action_space: number of discrete actions (e.g., turn left, forward, stop, etc.)
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.actor = nn.Linear(hidden_dim, action_space)  # policy logits
        self.critic = nn.Linear(hidden_dim, 1)            # state value for A2C
    
    def forward(self, fused_feat):
        """
        fused_feat: [B, input_dim] vision-language features
        Returns: action_logits [B, action_space], state_values [B,1].
        """
        x = self.relu(self.fc1(fused_feat))
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value