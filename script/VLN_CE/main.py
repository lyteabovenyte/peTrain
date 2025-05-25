import os
import json
import torch
import sys
import logging
from pathlib import Path
from torch.utils.data import DataLoader

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.VLN_CE.agent import VLNCEAgent
from models.VLN_CE.encoders import EncoderFactory
from script.VLN_CE.data_loader import VLNCEDataLoader
from script.VLN_CE.train_a2c import train_a2c
from script.VLN_CE.env import VLNCEEnv

def setup_logging(config):
    """Setup logging configuration."""
    log_dir = Path(config.get('log_dir', 'logs'))
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_config(config_path):
    """Load and validate configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Validate required sections
        required_sections = ['model', 'training', 'data_dir', 'train_split', 'val_split']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section '{section}' in config")
        
        # Validate model configuration
        model_config = config['model']
        required_model_sections = ['lang_encoder', 'visual_encoder', 'cross_modal', 'policy']
        for section in required_model_sections:
            if section not in model_config:
                raise ValueError(f"Missing required section '{section}' in model config")
        
        return config
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file: {e}")
    except Exception as e:
        raise ValueError(f"Error loading config: {e}")

def setup_device():
    """Setup and return the appropriate device."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")  # For M1 Mac
        logging.info("Using MPS (Metal Performance Shaders) device")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info("Using CUDA device")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU device")
    return device

def load_custom_encoders(config):
    """Load any custom encoder modules specified in the config."""
    # Load the default custom encoders module
    EncoderFactory.load_custom_encoders(['models.VLN_CE.custom_encoders'])
    logging.info("Loaded custom encoder modules")

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

def create_data_loader(config, split):
    """Create and return a data loader for the specified split."""
    supervision_type = config['training'].get('supervision_type', 'goal')
    dataset = VLNCEDataLoader(
        data_dir=config['data_dir'],
        split=split,
        max_instruction_length=config.get('max_instruction_length', 80),
        image_size=tuple(config['model']['visual_encoder'].get('input_size', [128, 128])),
        supervision_type=supervision_type
    )
    if supervision_type == 'LAW':
        return DataLoader(dataset, batch_size=config['training'].get('batch_size', 4), shuffle=True, collate_fn=law_collate_fn)
    else:
        return DataLoader(dataset, batch_size=config['training'].get('batch_size', 4), shuffle=True)

def main():
    # Load configuration
    config_path = 'configs/VLN-CE.json'
    config = load_config(config_path)
    
    # Setup logging
    logger = setup_logging(config)
    logger.info("Starting VLN-CE training")
    
    # Setup device
    device = setup_device()
    logger.info(f"Using device: {device}")
    
    # Load custom encoders
    logger.info("Loading custom encoders...")
    load_custom_encoders(config)
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader = create_data_loader(config, config['train_split'])
    val_loader = create_data_loader(config, config['val_split'])
    
    # Create agent
    logger.info("Creating agent...")
    agent = VLNCEAgent(config['model']).to(device)
    
    # Log model architecture
    logger.info("Model architecture:")
    logger.info(f"Language encoder: {config['model']['lang_encoder']['type']}")
    logger.info(f"Visual encoder: {config['model']['visual_encoder']['type']}")
    logger.info(f"Cross-modal transformer: {config['model']['cross_modal']['num_layers']} layers")
    
    # Create environment
    logger.info("Creating environment...")
    env = VLNCEEnv(config)
    
    # Start training
    logger.info("Starting training...")
    try:
        train_a2c(
            agent=agent,
            env=env,
            config=config['training'],
            train_loader=train_loader,
            val_loader=val_loader,
            device=device
        )
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        env.close()
        logger.info("Training finished")

if __name__ == "__main__":
    main() 