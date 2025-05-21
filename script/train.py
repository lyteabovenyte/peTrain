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
import argparse
import importlib.util

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from script.preprocessing import PointCloudDataset, PointCloudTransform, get_dataloaders

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cpu')  # Force CPU usage
        
        # Set number of workers based on CPU cores
        self.num_workers = min(2, multiprocessing.cpu_count())  # Reduced for CPU
        torch.set_num_threads(self.num_workers)  # Limit PyTorch threads
        
        self.setup_logging()
        self.setup_directories()
        
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
        
    def load_model_module(self, model_name):
        """Dynamically load a model module from the models directory"""
        models_dir = Path(__file__).parent.parent / 'models'
        model_path = models_dir / f"{model_name}.py"
        
        if not model_path.exists():
            raise ImportError(f"Model file not found: {model_path}")
            
        try:
            # Load the module dynamically
            spec = importlib.util.spec_from_file_location(model_name, model_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        except Exception as e:
            raise ImportError(f"Failed to load model {model_name}: {str(e)}")
        
    def setup_model(self):
        """Initialize model based on config"""
        try:
            # Get model name from config, default to PointNetPP if not specified
            model_name = self.config.get('model_name', 'PointNetPP')
            self.logger.info(f"Loading model: {model_name}")
            
            # Load the model module
            model_module = self.load_model_module(model_name)
            
            # Get the model class (assuming it has the same name as the file)
            model_class = getattr(model_module, model_name)
            
            # Initialize the model with config parameters
            model = model_class(
                num_classes=self.config['num_classes'],
                num_points=self.config['num_points']
            ).to(self.device)
            
            if self.config.get('pretrained', False) and self.config.get('pretrained_path'):
                pretrained_path = self.config['pretrained_path']
                if os.path.exists(pretrained_path):
                    self.logger.info(f"Loading pretrained weights from {pretrained_path}")
                    model.load_state_dict(torch.load(pretrained_path, map_location='cpu'))
                else:
                    self.logger.warning(f"Pretrained weights not found at {pretrained_path}")
            
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to setup model: {str(e)}")
            raise
    
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
        for batch_idx, (points, labels) in enumerate(pbar):
            # Move data to CPU and ensure contiguous memory
            points = points.contiguous()
            labels = labels.contiguous()
            
            # Get the most common label in each point cloud as the batch label
            batch_labels = torch.mode(labels, dim=1)[0]  # [B]
            
            optimizer.zero_grad()
            outputs = model(points)  # [B, num_classes]
            loss = criterion(outputs, batch_labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_labels.size(0)
            correct += predicted.eq(batch_labels).sum().item()
            
            # Clear memory
            del outputs, loss
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Calculate running averages
            avg_loss = total_loss / (batch_idx + 1)
            avg_acc = 100. * correct / total if total > 0 else 0
            
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'acc': f'{avg_acc:.2f}%'
            })
            
        # Calculate final averages
        avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
        avg_acc = 100. * correct / total if total > 0 else 0
        
        return avg_loss, avg_acc
    
    def validate(self, model, val_loader, criterion):
        """Validate the model"""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (points, labels) in enumerate(val_loader):
                # Move data to CPU and ensure contiguous memory
                points = points.contiguous()
                labels = labels.contiguous()
                
                # Get the most common label in each point cloud as the batch label
                batch_labels = torch.mode(labels, dim=1)[0]  # [B]
                
                outputs = model(points)  # [B, num_classes]
                loss = criterion(outputs, batch_labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += batch_labels.size(0)
                correct += predicted.eq(batch_labels).sum().item()
                
                # Clear memory
                del outputs, loss
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
        # Calculate final averages
        avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0
        avg_acc = 100. * correct / total if total > 0 else 0
                
        return avg_loss, avg_acc
    
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

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Training🏇...')
    parser.add_argument('--config', type=str, default='configs/PointNetPP.json',
                      help='Path to config file (relative to project root)')
    parser.add_argument('--model', type=str,
                      help='Model name to use (overrides config file setting)')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Initialize empty config
    config = {}
    
    # Load config from specified path
    config_path = os.path.join(str(Path(__file__).parent.parent), args.config)
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"Successfully loaded configuration from {config_path}")
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON format in {config_path}")
            print(f"Error details: {str(e)}")
            sys.exit(1)
    else:
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
    
    # Override model name if specified in command line
    if args.model:
        config['model_name'] = args.model
        print(f"Using model: {args.model} (overridden from command line)")
    
    # Initialize and start training
    trainer = Trainer(config)
    trainer.train()

if __name__ == '__main__':
    main()

