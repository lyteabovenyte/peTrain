import gymnasium as gym
import numpy as np
from gymnasium import spaces
import torch
import torchvision.transforms as T
from PIL import Image

class VLNCEEnv(gym.Env):
    """
    A custom environment for VLN-CE that implements the Gymnasium interface.
    This provides similar functionality to Habitat but with a simpler implementation.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Define action space (discrete actions)
        self.action_space = spaces.Discrete(6)  # 6 actions: forward, left, right, up, down, stop
        
        # Get max_instruction_length from the correct location in config
        max_instruction_length = config.get('max_instruction_length', 80)  # Default to 80 if not specified
        
        # Define observation space (RGB + depth)
        self.observation_space = spaces.Dict({
            'rgb': spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8),
            'depth': spaces.Box(low=0, high=1, shape=(224, 224, 1), dtype=np.float32),
            'instr_tokens': spaces.Box(low=0, high=config['vocab_size'], 
                                     shape=(max_instruction_length,), 
                                     dtype=np.int64)
        })
        
        # Image preprocessing
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize state
        self.current_position = None
        self.current_rotation = None
        self.goal_position = None
        self.goal_radius = None
        self.instruction = None
        self.instruction_tokens = None
        self.steps = 0
        self.max_steps = config.get('max_steps', 100)
        self.max_instruction_length = max_instruction_length
        
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Reset state
        self.current_position = np.array([0., 0., 0.])  # Start at origin
        self.current_rotation = np.array([0., 0., 0.])  # No rotation
        self.goal_position = np.array([10., 0., 10.])  # Example goal
        self.goal_radius = 3.0
        self.steps = 0
        
        # Generate dummy instruction tokens (replace with actual instruction processing)
        self.instruction_tokens = np.zeros(self.max_instruction_length, dtype=np.int64)
        
        # Generate initial observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action):
        """Execute one step in the environment."""
        self.steps += 1
        
        # Execute action (simplified movement)
        reward = self._execute_action(action)
        
        # Get new observation
        obs = self._get_observation()
        
        # Check if episode is done
        done = self._is_done()
        
        # Get info
        info = self._get_info()
        
        return obs, reward, done, False, info
    
    def _execute_action(self, action):
        """Execute the given action and return reward."""
        # Simplified movement logic
        if action == 0:  # forward
            self.current_position[0] += 0.5
        elif action == 1:  # left
            self.current_rotation[1] += np.pi/4
        elif action == 2:  # right
            self.current_rotation[1] -= np.pi/4
        elif action == 3:  # up
            self.current_position[1] += 0.5
        elif action == 4:  # down
            self.current_position[1] -= 0.5
        elif action == 5:  # stop
            pass
            
        # Calculate reward
        dist_to_goal = np.linalg.norm(self.current_position - self.goal_position)
        reward = -0.1  # Small negative reward for each step
        if dist_to_goal <= self.goal_radius:
            reward += 1.0  # Large reward for reaching goal
            
        return reward
    
    def _get_observation(self):
        """Generate current observation."""
        # Generate dummy RGB and depth images (replace with actual rendering)
        rgb = np.zeros((3, 224, 224), dtype=np.uint8)
        depth = np.zeros((1, 224, 224), dtype=np.float32)
        
        return {
            'rgb': rgb,
            'depth': depth,
            'instr_tokens': self.instruction_tokens
        }
    
    def _is_done(self):
        """Check if episode should end."""
        dist_to_goal = np.linalg.norm(self.current_position - self.goal_position)
        return (dist_to_goal <= self.goal_radius) or (self.steps >= self.max_steps)
    
    def _get_info(self):
        """Get additional information."""
        return {
            'distance_to_goal': np.linalg.norm(self.current_position - self.goal_position),
            'steps': self.steps,
            'success': np.linalg.norm(self.current_position - self.goal_position) <= self.goal_radius
        } 