import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.VLN_CE.encoders import EncoderFactory

# Custom collate function for LAW mode
def law_collate_fn(batch):
    batch_out = {}
    for key in batch[0]:
        if key == 'law_segments':
            batch_out[key] = [item[key] for item in batch]
        else:
            if isinstance(batch[0][key], torch.Tensor):
                batch_out[key] = torch.stack([item[key] for item in batch])
            else:
                batch_out[key] = [item[key] for item in batch]
    return batch_out

def train_dagger(agent, dataset, epochs=10, batch_size=4, lr=1e-4):
    """
    Trains the agent with DAgger imitation learning on the given dataset.
    """
    import json
    # Detect LAW mode from dataset (by presence of law_segments in first item)
    is_law = hasattr(dataset, 'supervision_type') and dataset.supervision_type == 'LAW'
    if is_law:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=law_collate_fn)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(agent.parameters(), lr=lr)
    
    agent.train()
    for epoch in range(epochs):
        for batch in loader:
            # Unpack batch
            instr_tokens = batch['instr_tokens']       # [B, L]
            gt_actions = batch['gt_actions']           # List of action lists or padded tensor
            # For simplicity, assume gt_actions is [B, T] tensor of action indices
            # Also assume environment resets and runs internally or actions given externally.
            
            # Simulate a single step for illustration:
            # Forward pass: (we skip environment observation; assume visual input is available)
            # Normally, you would get current observation from Habitat here:
            # rgb, depth = env.get_observation()
            # For CPU-friendly pseudo-training, we can use zeros:
            B = instr_tokens.size(0)
            dummy_rgb = torch.zeros((B,3,128,128))   # placeholder image batch
            dummy_depth = torch.zeros((B,1,128,128))
            
            logits, _ = agent(dummy_rgb, dummy_depth, instr_tokens)
            # Compute cross-entropy loss against ground-truth actions (teacher forcing)
            # We assume gt_actions contains the next-step action indices for each sample
            action_targets = torch.LongTensor(gt_actions)  # [B] or [B,1]
            loss = F.cross_entropy(logits, action_targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch}: imitation loss = {loss.item():.4f}")

if __name__ == "__main__":
    import sys
    import os
    import json
    from models.VLN_CE.agent import VLNCEAgent
    from script.VLN_CE.data_loader import VLNCEDataLoader
    from models.VLN_CE.encoders import EncoderFactory

    # Use a debug config or fallback to default
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/VLN-CE.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Register custom encoders (e.g., mobilenet)
    EncoderFactory.load_custom_encoders(['models.VLN_CE.custom_encoders'])

    # Load agent and dataset (limit to 5 episodes for fast debug)
    agent = VLNCEAgent(config['model'])
    dataset = VLNCEDataLoader(
        data_dir=config['data_dir'],
        split=config['train_split'],
        max_instruction_length=config.get('max_instruction_length', 80),
        image_size=tuple(config['model']['visual_encoder'].get('input_size', [128, 128])),
        limit=5
    )

    print("[DAgger Debug] Training on 5 episodes for 2 epochs...")
    train_dagger(agent, dataset, epochs=2, batch_size=2, lr=1e-4)
    print("[DAgger Debug] Done.")