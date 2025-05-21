import os
import json
import torch
import sys

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.VLN_CE.agent import VLNCEAgent
from script.VLN_CE.data_loader import VLNCEDataLoader
from script.VLN_CE.train_a2c import train_a2c
from script.VLN_CE.env import VLNCEEnv

def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def main():
    # Load configuration
    config = load_config('configs/VLN-CE.json')
    
    # Set device
    device = torch.device('cpu')  # Using CPU for M1 MacBook
    
    # Create data loader
    print("Creating data loader...")
    data_loader = VLNCEDataLoader(
        data_dir=config['data_dir'],
        split=config['train_split'],
        max_instruction_length=config['model']['max_instruction_length'],
        image_size=tuple(config['model']['input_size'])
    )
    
    # Create agent
    print("Creating agent...")
    agent = VLNCEAgent(
        vocab_size=config['vocab_size'],
        embed_dim=config['embed_dim'],
        lang_hidden=config['lang_hidden'],
        visual_dim=config['visual_dim'],
        attn_hidden=config['attn_hidden'],
        policy_hidden=config['policy_hidden'],
        action_space=config['action_space']
    ).to(device)
    
    # Create environment
    print("Creating environment...")
    env = VLNCEEnv(config)
    
    # Start training
    print("Starting training...")
    train_a2c(agent, env, config['training'])
    
    # Cleanup
    env.close()

if __name__ == "__main__":
    main() 