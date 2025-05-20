"""
    Load point cloud dataset from data/processed/.
    Choose architecture dynamically via configs.
    Save weights to data/models/.
    CPU-optimized version.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import logging
import json
from datetime import datetime
import sys
from pathlib import Path
import multiprocessing

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from script.preprocessing import get_dataloaders
from models.pointnetpp import PointNetPP

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cpu')  # Force CPU usage
        self.setup_logging()
        self.setup_directories()
        
        # Set number of workers based on CPU cores
        self.num_workers = min(4, multiprocessing.cpu_count())
        torch.set_num_threads(self.num_workers)  # Limit PyTorch threads
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = os.path.join(self.config['output_dir'], 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'training_{timestamp}.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Using {self.num_workers} workers for data loading")
        
    def setup_directories(self):
        """Create necessary directories"""
        os.makedirs(self.config['output_dir'], exist_ok=True)
        os.makedirs(self.config['model_dir'], exist_ok=True)
        
    def setup_model(self):
        """Initialize model based on config"""
        model = PointNetPP(
            num_classes=self.config['num_classes'],
            num_points=self.config['num_points']
        ).to(self.device)
        
        #! pretrained models and tune config file
        if self.config['pretrained'] and os.path.exists(self.config['pretrained_path']):
            self.logger.info(f"Loading pretrained weights from {self.config['pretrained_path']}")
            model.load_state_dict(torch.load(self.config['pretrained_path'], map_location='cpu'))
            
        return model
    
    def setup_optimizer(self, model):
        """Setup optimizer and learning rate scheduler"""
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        return optimizer, scheduler
    
    def train_epoch(self, model, train_loader, optimizer, criterion):
        """Train for one epoch"""
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for points, labels in pbar:
            # Move data to CPU and ensure contiguous memory
            points = points.contiguous()
            labels = labels.contiguous()
            
            optimizer.zero_grad()
            outputs = model(points)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Clear memory
            # del outputs, loss
            # torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            pbar.set_postfix({
                'loss': f'{total_loss/pbar.n:.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
            
        return total_loss / len(train_loader), 100. * correct / total
    
    def validate(self, model, val_loader, criterion):
        """Validate the model"""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for points, labels in tqdm(val_loader, desc='Validation'):
                # Move data to CPU and ensure contiguous memory
                points = points.contiguous()
                labels = labels.contiguous()
                
                outputs = model(points)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Clear memory
                del outputs, loss
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
        return total_loss / len(val_loader), 100. * correct / total
    
    def save_checkpoint(self, model, optimizer, epoch, metrics):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }
        
        checkpoint_path = os.path.join(
            self.config['model_dir'],
            f'checkpoint_epoch_{epoch}.pt'
        )
        torch.save(checkpoint, checkpoint_path, _use_new_zipfile_serialization=False)
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")
        
    def train(self):
        """Main training loop"""
        self.logger.info("Starting training...")
        self.logger.info(f"Using device: {self.device}")
        
        # Setup data with CPU-friendly settings
        train_loader, val_loader = get_dataloaders(
            self.config['data_dir'],
            batch_size=self.config['batch_size'],
            num_points=self.config['num_points']
        )
        
        # Setup model and training components
        model = self.setup_model()
        optimizer, scheduler = self.setup_optimizer(model)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        best_val_acc = 0
        for epoch in range(self.config['epochs']):
            self.logger.info(f"\nEpoch {epoch+1}/{self.config['epochs']}")
            
            # Train
            train_loss, train_acc = self.train_epoch(
                model, train_loader, optimizer, criterion
            )
            
            # Validate
            val_loss, val_acc = self.validate(model, val_loader, criterion)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Log metrics
            metrics = {
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc
            }
            self.logger.info(
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )
            
            # Save checkpoint
            self.save_checkpoint(model, optimizer, epoch, metrics)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_path = os.path.join(
                    self.config['model_dir'],
                    'best_model.pt'
                )
                torch.save(model.state_dict(), best_model_path, _use_new_zipfile_serialization=False)
                self.logger.info(f"New best model saved with accuracy: {val_acc:.2f}%")

def main():
    # Default configuration optimized for CPU
    config = {
        'data_dir': '../data/processed',
        'output_dir': '../data/output',
        'model_dir': '../data/models',
        'num_classes': 10,  # Update based on your dataset
        'num_points': 1024,
        'batch_size': 8,  # Reduced batch size for CPU
        'learning_rate': 0.0005,  # Slightly reduced learning rate
        'weight_decay': 1e-4,
        'epochs': 100,
        'pretrained': False,
        'pretrained_path': None
    }
    
    # Load custom config if exists
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            custom_config = json.load(f)
            config.update(custom_config)
    
    # Initialize and start training
    trainer = Trainer(config)
    trainer.train()

if __name__ == '__main__':
    main()

