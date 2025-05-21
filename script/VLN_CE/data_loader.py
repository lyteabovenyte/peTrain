import os
import gzip
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class VLNCEDataLoader(Dataset):
    """
    Loads VLN-CE episodes from a JSON.gz file with proper error handling and preprocessing.
    Each episode includes instruction tokens, start pose, goal, and reference path.
    If ground-truth actions are available (e.g., in *_follower_gt.json.gz), they can be loaded as well.
    """
    def __init__(self, data_dir, split, max_instruction_length=80, image_size=(128, 128)):
        """
        data_dir: root directory for VLN-CE data (e.g., ../data/VLN-CE/R2R_VLNCE_v1-3_preprocessed/train)
        split: 'train', 'val_seen', or 'val_unseen'
        max_instruction_length: maximum number of tokens in instructions
        image_size: target size for RGB images
        """
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.max_instruction_length = max_instruction_length
        self.image_size = image_size
        
        # Simplified image preprocessing for CPU
        self.transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Validate data directory
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
            
        # Try both .json and .json.gz files
        epi_file = os.path.join(data_dir, f"{split}.json")
        epi_file_gz = os.path.join(data_dir, f"{split}.json.gz")
        
        # Load episodes with error handling
        try:
            if os.path.exists(epi_file_gz):
                with gzip.open(epi_file_gz, 'rt') as f:
                    data = json.load(f)
            elif os.path.exists(epi_file):
                with open(epi_file, 'r') as f:
                    data = json.load(f)
            else:
                raise FileNotFoundError(f"No data file found at {epi_file} or {epi_file_gz}")
            self.episodes = data['episodes']
        except Exception as e:
            raise RuntimeError(f"Failed to load episodes: {str(e)}")
            
        # Load ground truth if available for imitation learning
        self.gt_actions = {}
        gt_file = os.path.join(data_dir, f"{split}_follower_gt.json.gz")
        if os.path.exists(gt_file):
            try:
                with gzip.open(gt_file, 'rt') as f:
                    gt_data = json.load(f)
                self.gt_actions = {e['episode_id']: e['actions'] 
                                 for e in gt_data['episodes']}
            except Exception as e:
                print(f"Warning: Failed to load ground truth from {gt_file}: {str(e)}")
                
        # Validate episodes
        self._validate_episodes()
        
        # Cache for token indices to reduce computation
        self.token_cache = {}
        
    def _validate_episodes(self):
        """Validate episode data structure and content."""
        required_fields = ['episode_id', 'instruction', 'path', 'scan', 'start_viewpoint', 'end_viewpoint']
        for ep in self.episodes:
            for field in required_fields:
                if field not in ep:
                    raise ValueError(f"Missing required field '{field}' in episode {ep.get('episode_id', 'unknown')}")
            if not ep['path']:
                raise ValueError(f"No path found in episode {ep['episode_id']}")
                
    def _preprocess_instruction(self, instruction):
        """Preprocess instruction text into tokens."""
        # Simple tokenization (split by space)
        tokens = instruction.lower().split()
        # Pad or truncate to max_instruction_length
        if len(tokens) > self.max_instruction_length:
            tokens = tokens[:self.max_instruction_length]
        else:
            tokens = tokens + ['<pad>'] * (self.max_instruction_length - len(tokens))
        # Convert tokens to indices (simple mapping for now)
        token_to_idx = {token: idx for idx, token in enumerate(set(tokens))}
        indices = [token_to_idx.get(token, 0) for token in tokens]
        return torch.LongTensor(indices)
        
    def _load_image(self, image_path):
        """Load and preprocess image with error handling."""
        try:
            image = Image.open(image_path).convert('RGB')
            return self.transform(image)
        except Exception as e:
            print(f"Warning: Failed to load image {image_path}: {str(e)}")
            # Return a zero tensor as fallback
            return torch.zeros(3, *self.image_size)
            
    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        try:
            ep = self.episodes[idx]
            
            # Preprocess instruction
            instr_tokens = self._preprocess_instruction(ep['instruction'])
            
            # Extract path information
            path = ep['path']
            start_pos = torch.tensor([path[0]['heading'], path[0]['elevation']], dtype=torch.float32)
            end_pos = torch.tensor([path[-1]['heading'], path[-1]['elevation']], dtype=torch.float32)
            
            # For now, return dummy images since we don't have actual image data
            # Ensure CHW format for PyTorch
            rgb = torch.zeros(3, *self.image_size)  # [3, H, W]
            depth = torch.zeros(1, *self.image_size)  # [1, H, W]
            
            # Get ground-truth actions if available
            actions = torch.tensor(self.gt_actions.get(ep['episode_id'], []), dtype=torch.long)
            
            return {
                'episode_id': ep['episode_id'],
                'instr_tokens': instr_tokens,
                'rgb': rgb,
                'depth': depth,
                'start_pos': start_pos,
                'end_pos': end_pos,
                'scan': ep['scan'],
                'gt_actions': actions
            }
            
        except Exception as e:
            print(f"Error loading episode {idx}: {str(e)}")
            # Return a default item or re-raise the exception
            raise