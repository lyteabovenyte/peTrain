import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from tqdm import tqdm
import os
import json
from datetime import datetime

class A2CTrainer:
    def __init__(self, agent, env, config):
        self.agent = agent
        self.env = env
        self.config = config
        
        # Training hyperparameters
        self.gamma = config.get('gamma', 0.99)
        self.lr = config.get('lr', 1e-4)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.value_coef = config.get('value_coef', 0.5)
        self.max_episodes = config.get('max_episodes', 1000)
        self.save_interval = config.get('save_interval', 100)
        
        # Initialize optimizer with CPU-friendly settings
        self.optimizer = torch.optim.Adam(
            self.agent.parameters(),
            lr=self.lr,
            eps=1e-5,
            weight_decay=1e-4
        )
        
        # Create checkpoint directory
        self.checkpoint_dir = os.path.join('checkpoints', datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Save config
        with open(os.path.join(self.checkpoint_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
            
        # Training metrics
        self.metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'success_rates': [],
            'losses': []
        }
        
    def compute_returns_and_advantages(self, rewards, values, dones):
        """Compute returns and advantages using GAE."""
        returns = []
        advantages = []
        R = 0
        A = 0
        for r, v, d in zip(reversed(rewards), reversed(values), reversed(dones)):
            R = r + self.gamma * R * (1 - d)
            returns.insert(0, R)
            td_error = r + self.gamma * v * (1 - d) - v
            A = td_error + self.gamma * self.gamma * A * (1 - d)
            advantages.insert(0, A)
        return torch.tensor(returns), torch.tensor(advantages)
    
    def save_checkpoint(self, episode, metrics):
        """Save model checkpoint and training metrics."""
        checkpoint = {
            'episode': episode,
            'model_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }
        path = os.path.join(self.checkpoint_dir, f'checkpoint_ep{episode}.pt')
        torch.save(checkpoint, path)
        
        # Save metrics separately for easier plotting
        metrics_path = os.path.join(self.checkpoint_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def train(self):
        """Main training loop with CPU optimizations."""
        self.agent.train()
        
        for episode in tqdm(range(self.max_episodes)):
            # Reset environment
            obs, _ = self.env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            # Episode storage
            log_probs = []
            values = []
            rewards = []
            dones = []
            
            while not done:
                # Preprocess observations and ensure they require gradients
                rgb = torch.from_numpy(obs['rgb']).float().unsqueeze(0).requires_grad_(True)
                depth = torch.from_numpy(obs['depth']).float().unsqueeze(0).requires_grad_(True)
                instr = torch.LongTensor(obs['instr_tokens']).unsqueeze(0)
                
                # Forward pass
                logits, value = self.agent(rgb, depth, instr)
                
                # Sample action
                probs = F.softmax(logits, dim=-1)
                dist = Categorical(probs)
                action = dist.sample()
                
                # Store values
                log_probs.append(dist.log_prob(action))
                values.append(value)
                
                # Step environment
                obs, reward, terminated, truncated, info = self.env.step(action.item())
                done = terminated or truncated
                
                # Store transition
                rewards.append(reward)
                dones.append(done)
                episode_reward += reward
                episode_length += 1
                
                # Clear memory periodically
                if episode_length % 10 == 0:
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Compute returns and advantages
            returns, advantages = self.compute_returns_and_advantages(
                rewards, values, dones
            )
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Convert to tensors and ensure they require gradients
            log_probs = torch.cat(log_probs).requires_grad_(True)
            values = torch.cat(values).requires_grad_(True)
            
            # Compute losses
            policy_loss = -(log_probs * advantages).mean()
            value_loss = F.mse_loss(values, returns)
            entropy_loss = -dist.entropy().mean()
            
            # Total loss
            loss = (policy_loss + 
                   self.value_coef * value_loss - 
                   self.entropy_coef * entropy_loss)
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.agent.parameters(), 
                self.max_grad_norm
            )
            
            self.optimizer.step()
            
            # Update metrics
            self.metrics['episode_rewards'].append(episode_reward)
            self.metrics['episode_lengths'].append(episode_length)
            self.metrics['success_rates'].append(float(info.get('success', False)))
            self.metrics['losses'].append(loss.item())
            
            # Print progress
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.metrics['episode_rewards'][-10:])
                avg_length = np.mean(self.metrics['episode_lengths'][-10:])
                success_rate = np.mean(self.metrics['success_rates'][-10:])
                print(f"\nEpisode {episode + 1}")
                print(f"Average Reward: {avg_reward:.2f}")
                print(f"Average Length: {avg_length:.2f}")
                print(f"Success Rate: {success_rate:.2%}")
                print(f"Loss: {loss.item():.4f}")
            
            # Save checkpoint
            if (episode + 1) % self.save_interval == 0:
                self.save_checkpoint(episode + 1, self.metrics)
                
            # Clear memory
            del log_probs, values, rewards, dones, returns, advantages
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

def train_a2c(agent, env, config):
    """Wrapper function to start training."""
    trainer = A2CTrainer(agent, env, config)
    trainer.train()