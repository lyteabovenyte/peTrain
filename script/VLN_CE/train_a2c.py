import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from tqdm import tqdm
import os
import json
from datetime import datetime

class A2CTrainer:
    def __init__(self, agent, env, config, train_loader=None, val_loader=None, device=None):
        self.agent = agent
        self.env = env
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Training hyperparameters
        self.gamma = config.get('gamma', 0.99)
        self.lr = config.get('lr', 5e-5)
        self.max_grad_norm = config.get('max_grad_norm', 1.0)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.value_coef = config.get('value_coef', 0.5)
        self.max_episodes = config.get('max_episodes', 1000)
        self.save_interval = config.get('save_interval', 100)
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 2)
        
        # Initialize optimizer with better settings
        self.optimizer = torch.optim.AdamW(
            self.agent.parameters(),
            lr=self.lr,
            eps=config.get('adam_eps', 1e-8),
            betas=(config.get('adam_beta1', 0.9), config.get('adam_beta2', 0.999)),
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.get('lr_scheduler', {}).get('max_steps', 10000),
            T_mult=1,
            eta_min=1e-6
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
            'losses': [],
            'learning_rates': []
        }
        
        # Move model to device
        self.agent = self.agent.to(self.device)
        
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
        return torch.tensor(returns, device=self.device), torch.tensor(advantages, device=self.device)
    
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
        """Main training loop with improved stability."""
        self.agent.train()
        global_step = 0
        
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
                # Preprocess observations and move to device
                rgb = torch.from_numpy(obs['rgb']).float().unsqueeze(0).to(self.device)
                depth = torch.from_numpy(obs['depth']).float().unsqueeze(0).to(self.device)
                instr = torch.LongTensor(obs['instr_tokens']).unsqueeze(0).to(self.device)
                
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
                
                # Update learning rate
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
                self.metrics['learning_rates'].append(current_lr)
                
                # Gradient accumulation
                if episode_length % self.gradient_accumulation_steps == 0:
                    # Compute returns and advantages
                    returns, advantages = self.compute_returns_and_advantages(
                        rewards[-self.gradient_accumulation_steps:],
                        values[-self.gradient_accumulation_steps:],
                        dones[-self.gradient_accumulation_steps:]
                    )
                    
                    # Normalize advantages
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                    
                    # Convert to tensors and move to device
                    log_probs_batch = torch.cat(log_probs[-self.gradient_accumulation_steps:]).to(self.device)
                    values_batch = torch.cat(values[-self.gradient_accumulation_steps:]).to(self.device)
                    
                    # Compute losses
                    policy_loss = -(log_probs_batch * advantages).mean()
                    # Ensure returns and values have the same shape
                    returns = returns.view(-1, 1)  # [B, 1]
                    values_batch = values_batch.view(-1, 1)  # [B, 1]
                    value_loss = F.mse_loss(values_batch, returns)
                    entropy_loss = -dist.entropy().mean()
                    
                    # Total loss
                    loss = (policy_loss + 
                           self.value_coef * value_loss - 
                           self.entropy_coef * entropy_loss)
                    
                    # Scale loss for gradient accumulation
                    loss = loss / self.gradient_accumulation_steps
                    
                    # Backward pass
                    loss.backward()
                    
                    # Update metrics
                    self.metrics['losses'].append(loss.item() * self.gradient_accumulation_steps)
                    
                    # Optimize
                    if (episode_length + 1) % self.gradient_accumulation_steps == 0:
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(
                            self.agent.parameters(), 
                            self.max_grad_norm
                        )
                        self.optimizer.step()
                        self.optimizer.zero_grad()
            
            # Update metrics
            self.metrics['episode_rewards'].append(episode_reward)
            self.metrics['episode_lengths'].append(episode_length)
            self.metrics['success_rates'].append(float(info.get('success', False)))
            
            # Print progress
            if (episode + 1) % 2 == 0:
                avg_reward = np.mean(self.metrics['episode_rewards'][-2:])
                avg_length = np.mean(self.metrics['episode_lengths'][-2:])
                success_rate = np.mean(self.metrics['success_rates'][-2:])
                avg_loss = np.mean(self.metrics['losses'][-2:])
                print(f"\nEpisode {episode + 1}")
                print(f"Average Reward: {avg_reward:.2f}")
                print(f"Average Length: {avg_length:.2f}")
                print(f"Success Rate: {success_rate:.2%}")
                print(f"Loss: {avg_loss:.4f}")
                print(f"Learning Rate: {current_lr:.2e}")
            
            # Save checkpoint
            if (episode + 1) % self.save_interval == 0:
                self.save_checkpoint(episode + 1, self.metrics)

def train_a2c(agent, env, config, train_loader=None, val_loader=None, device=None):
    """Wrapper function to start training."""
    trainer = A2CTrainer(agent, env, config, train_loader, val_loader, device)
    trainer.train()