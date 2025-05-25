import os
import gzip
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import math

class VLNCEDataLoader(Dataset):
    """
    Loads VLN-CE episodes from a JSON.gz file with proper error handling and preprocessing.
    Each episode includes instruction tokens, start pose, goal, and reference path.
    If ground-truth actions are available, they can be loaded as well.
    """
    def __init__(self, data_dir, split, max_instruction_length=80, image_size=(128, 128), limit=None, supervision_type="goal"):
        """
        data_dir: root directory for VLN-CE data (e.g., ../data/VLN-CE/R2R_VLNCE_v1-3_preprocessed/)
        split: 'train', 'val_seen', or 'val_unseen'
        max_instruction_length: maximum number of tokens in instructions
        image_size: target size for RGB images
        limit: if not None, only load the first 'limit' episodes (for debugging)
        supervision_type: type of supervision to use ('goal' or 'LAW')
        """
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.max_instruction_length = max_instruction_length
        self.image_size = image_size
        self.supervision_type = supervision_type
        
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
            
    def _split_sub_instructions(self, instruction_text):
        """
        Split the instruction into sub-instructions using the word 'and'.
        Returns a list of sub-instructions (stripped of whitespace).
        """
        if not isinstance(instruction_text, str):
            return [""]
        return [s.strip() for s in instruction_text.split('and') if s.strip()]

    def _angle_between(self, v1, v2):
        """Compute signed angle (in radians) between two 2D vectors."""
        x1, z1 = v1[0], v1[2]
        x2, z2 = v2[0], v2[2]
        dot = x1 * x2 + z1 * z2
        det = x1 * z2 - z1 * x2
        angle = math.atan2(det, dot)
        return angle

    def _direction_vector(self, p_from, p_to):
        """Get normalized 2D direction vector from p_from to p_to."""
        dx = p_to[0] - p_from[0]
        dz = p_to[2] - p_from[2]
        norm = math.sqrt(dx * dx + dz * dz) + 1e-8
        return [dx / norm, dz / norm]

    def _generate_advanced_law_segments(self, ep):
        """
        For Advanced LAW: align sub-instructions to reference_path segments.
        Returns a list of dicts: [{sub_instruction, action, waypoint}]
        """
        instruction_text = ep.get('instruction', {}).get('instruction_text', "")
        sub_instructions = self._split_sub_instructions(instruction_text)
        ref_path = ep.get('reference_path', [])
        if not ref_path or len(ref_path) < 2 or not sub_instructions:
            return []
        num_segments = len(ref_path) - 1
        seg_per_sub = max(1, num_segments // len(sub_instructions))
        law_segments = []
        seg_idx = 0
        # Initial heading: direction from first to second waypoint
        if num_segments > 0:
            heading = self._direction_vector(ref_path[0], ref_path[1])
        else:
            heading = [1.0, 0.0]  # Arbitrary
        for i, sub_instr in enumerate(sub_instructions):
            if i == len(sub_instructions) - 1:
                seg_end = num_segments
            else:
                seg_end = min(num_segments, seg_idx + seg_per_sub)
            for j in range(seg_idx, seg_end):
                curr = ref_path[j]
                nxt = ref_path[j + 1]
                target_dir = self._direction_vector(curr, nxt)
                angle = self._angle_between(heading, target_dir)
                # Discretize angle: threshold for left/right/forward
                actions = []
                angle_deg = math.degrees(angle)
                if abs(angle_deg) < 15:
                    actions.append(0)  # forward
                else:
                    # Turn left (1) or right (2) as needed, then forward
                    if angle_deg > 0:
                        actions.extend([1] * int(abs(angle_deg) // 30))  # left
                    else:
                        actions.extend([2] * int(abs(angle_deg) // 30))  # right
                    actions.append(0)  # forward after turning
                # For each action, add a law_segment
                for act in actions:
                    law_segments.append({
                        'sub_instruction': sub_instr,
                        'action': act,
                        'waypoint': nxt
                    })
                # Update heading to new direction
                heading = target_dir
            seg_idx = seg_end
        # Add stop action at the end
        law_segments.append({
            'sub_instruction': sub_instructions[-1] if sub_instructions else "",
            'action': 5,  # stop
            'waypoint': ref_path[-1]
        })
        return law_segments
        
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
            
            # Get supervision actions
            if self.supervision_type == "LAW":
                law_segments = self._generate_advanced_law_segments(ep)
            else:
                actions = torch.tensor(self.gt_actions.get(str(ep['episode_id']), []), dtype=torch.long)
            
            result = {
                'episode_id': ep['episode_id'],
                'instr_tokens': instr_tokens,
                'rgb': rgb,
                'depth': depth,
                'start_pos': start_pos,
                'end_pos': end_pos,
                'scan': ep['scene_id']
            }
            if self.supervision_type == "LAW":
                result['law_segments'] = law_segments
            else:
                result['gt_actions'] = actions
            return result
            
        except Exception as e:
            print(f"Error loading episode {idx}: {str(e)}")
            # Return a default item or re-raise the exception
            raise