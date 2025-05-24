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
    If ground-truth actions are available, they can be loaded as well.
    """
    def __init__(self, data_dir, split, max_instruction_length=80, image_size=(128, 128), limit=None):
        """
        data_dir: root directory for VLN-CE data (e.g., ../data/VLN-CE/R2R_VLNCE_v1-3_preprocessed/)
        split: 'train', 'val_seen', or 'val_unseen'
        max_instruction_length: maximum number of tokens in instructions
        image_size: target size for RGB images
        limit: if not None, only load the first 'limit' episodes (for debugging)
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
            
        # Construct path to split directory
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
            
        # Try both .json and .json.gz files in the split directory
        epi_file = os.path.join(split_dir, f"{split}.json")
        epi_file_gz = os.path.join(split_dir, f"{split}.json.gz")
        
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
            
            # Handle different data formats
            if isinstance(data, dict):
                # If data is a dictionary, try to extract episodes
                if 'episodes' in data:
                    self.episodes = data['episodes']
                elif 'data' in data:
                    self.episodes = data['data']
                else:
                    # If no clear episodes field, convert dict to list
                    self.episodes = [{'episode_id': k, **v} for k, v in data.items()]
            elif isinstance(data, list):
                self.episodes = data
            else:
                raise ValueError(f"Unexpected data format: {type(data)}")
                
            # Ensure episodes is a list of dictionaries
            if not isinstance(self.episodes, list):
                raise ValueError(f"Expected list of episodes, got {type(self.episodes)}")
                
            # Validate each episode is a dictionary
            for i, ep in enumerate(self.episodes):
                if not isinstance(ep, dict):
                    raise ValueError(f"Episode at index {i} is not a dictionary")
                
        except Exception as e:
            raise RuntimeError(f"Failed to load episodes: {str(e)}")
            
        # Load ground truth if available for imitation learning
        self.gt_actions = {}
        gt_file = os.path.join(split_dir, f"{split}_gt.json.gz")
        if os.path.exists(gt_file):
            try:
                with gzip.open(gt_file, 'rt') as f:
                    gt_data = json.load(f)
                if isinstance(gt_data, list):
                    self.gt_actions = {str(e.get('episode_id', '')): e.get('actions', []) 
                                     for e in gt_data if isinstance(e, dict)}
                elif isinstance(gt_data, dict):
                    self.gt_actions = {str(k): v.get('actions', []) 
                                     for k, v in gt_data.items()}
            except Exception as e:
                print(f"Warning: Failed to load ground truth from {gt_file}: {str(e)}")
                
        # Validate episodes
        self._validate_episodes()
        
        # Subsample episodes if limit is set
        if limit is not None:
            self.episodes = self.episodes[:limit]
        
        # Cache for token indices to reduce computation
        self.token_cache = {}
        
    def _validate_episodes(self):
        """Validate episode data structure and content."""
        required_fields = [
            'episode_id', 'scene_id', 'start_position', 'start_rotation',
            'goals', 'instruction', 'reference_path'
        ]
        for i, ep in enumerate(self.episodes):
            if not isinstance(ep, dict):
                raise ValueError(f"Episode at index {i} is not a dictionary")
            for field in required_fields:
                if field not in ep:
                    raise ValueError(f"Missing required field '{field}' in episode {ep.get('episode_id', f'at index {i}')}")
            if not ep['reference_path']:
                raise ValueError(f"No reference path found in episode {ep.get('episode_id', f'at index {i}')}")
                
    def _preprocess_instruction(self, instruction):
        """Preprocess instruction text into tokens."""
        # Use pre-tokenized instruction tokens if available
        if isinstance(instruction, dict) and 'instruction_tokens' in instruction:
            tokens = instruction['instruction_tokens']
            # Pad or truncate to max_instruction_length
            if len(tokens) > self.max_instruction_length:
                tokens = tokens[:self.max_instruction_length]
            else:
                tokens = tokens + [0] * (self.max_instruction_length - len(tokens))
            return torch.LongTensor(tokens)
        else:
            # Fallback to text tokenization if needed
            text = instruction['instruction_text'] if isinstance(instruction, dict) else instruction
            tokens = text.lower().split()
            if len(tokens) > self.max_instruction_length:
                tokens = tokens[:self.max_instruction_length]
            else:
                tokens = tokens + ['<pad>'] * (self.max_instruction_length - len(tokens))
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
            start_pos = torch.tensor(ep['start_position'], dtype=torch.float32)
            end_pos = torch.tensor(ep['goals'][0]['position'], dtype=torch.float32)
            
            # For now, return dummy images since we don't have actual image data
            # Ensure CHW format for PyTorch
            rgb = torch.zeros(3, *self.image_size)  # [3, H, W]
            depth = torch.zeros(1, *self.image_size)  # [1, H, W]
            
            # Get ground-truth actions if available
            actions = torch.tensor(self.gt_actions.get(str(ep['episode_id']), []), dtype=torch.long)
            
            return {
                'episode_id': ep['episode_id'],
                'instr_tokens': instr_tokens,
                'rgb': rgb,
                'depth': depth,
                'start_pos': start_pos,
                'end_pos': end_pos,
                'scan': ep['scene_id'],
                'gt_actions': actions
            }
            
        except Exception as e:
            print(f"Error loading episode {idx}: {str(e)}")
            # Return a default item or re-raise the exception
            raise