"""
Supervised Learning Trainer - Simplified PyTorch implementation

Trains transformer model using pre-generated supervised learning data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import time
from typing import Dict, Any, Optional
from pathlib import Path

from .transformer_model import TransformerPredator
from .simulation_dataset import SimulationDataset, create_dataloader, load_train_val_datasets

class SupervisedTrainer:
    """Simplified supervised learning trainer"""
    
    def __init__(self,
                 model: TransformerPredator,
                 train_dataset: SimulationDataset,
                 val_dataset: SimulationDataset,
                 device: str = 'auto',
                 learning_rate: float = 1e-3,
                 batch_size: int = 32,
                 weight_decay: float = 1e-4,
                 gradient_clip: float = 1.0,
                 log_dir: str = 'runs',
                 checkpoint_dir: str = 'checkpoints'):
        
        # Device setup
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Training on device: {self.device}")
        
        # Model setup
        self.model = model.to(self.device)
        
        # Dataset setup
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        # Create data loaders
        self.train_loader = create_dataloader(train_dataset, batch_size, shuffle=True)
        self.val_loader = create_dataloader(val_dataset, batch_size, shuffle=False)
        
        # Training setup
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.criterion = nn.MSELoss()
        self.gradient_clip = gradient_clip
        
        # Logging setup
        self.writer = SummaryWriter(log_dir)
        
        # Checkpoint setup
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        print(f"Trainer initialized:")
        print(f"  Model parameters: {self.model.get_num_parameters():,}")
        print(f"  Training samples: {len(train_dataset):,}")
        print(f"  Validation samples: {len(val_dataset):,}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {learning_rate}")
    
    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        for batch_idx, (batch_inputs, batch_targets) in enumerate(self.train_loader):
            # Move targets to device
            batch_targets = batch_targets.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(batch_inputs)
            loss = self.criterion(predictions, batch_targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            
            # Optimizer step
            self.optimizer.step()
            
            # Logging
            total_loss += loss.item()
            self.global_step += 1
            
            # Log to tensorboard
            if batch_idx % 10 == 0:
                self.writer.add_scalar('Train/BatchLoss', loss.item(), self.global_step)
            
            # Progress logging
            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx}/{num_batches}, Loss: {loss.item():.6f}")
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self) -> float:
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch_inputs, batch_targets in self.val_loader:
                batch_targets = batch_targets.to(self.device)
                
                predictions = self.model(batch_inputs)
                loss = self.criterion(predictions, batch_targets)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def save_checkpoint(self, filename: str = None, is_best: bool = False):
        """Save model checkpoint"""
        if filename is None:
            filename = f'checkpoint_epoch_{self.epoch}.pt'
        
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'global_step': self.global_step
        }
        
        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"  Saved best model to {best_path}")
        
        return filepath
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.global_step = checkpoint['global_step']
        
        print(f"Loaded checkpoint from epoch {self.epoch}")
    
    def train(self, num_epochs: int, early_stopping_patience: int = 20):
        """Main training loop"""
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Early stopping patience: {early_stopping_patience}")
        print(f"Saving checkpoint every epoch to: {self.checkpoint_dir}")
        
        early_stopping_counter = 0
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            epoch_start_time = time.time()
            
            # Training
            train_loss = self.train_epoch()
            
            # Validation
            val_loss = self.validate()
            
            # Logging
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss: {val_loss:.6f}")
            
            # TensorBoard logging
            self.writer.add_scalar('Train/EpochLoss', train_loss, epoch)
            self.writer.add_scalar('Val/EpochLoss', val_loss, epoch)
            
            # Check for improvement
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                early_stopping_counter = 0
                print(f"  New best validation loss: {val_loss:.6f}")
            else:
                early_stopping_counter += 1
            
            # Save checkpoint every epoch (not just when best)
            checkpoint_file = f'checkpoint_epoch_{epoch+1}.pt'
            self.save_checkpoint(checkpoint_file, is_best=is_best)
            print(f"  Saved checkpoint: {checkpoint_file}")
            
            # Early stopping
            if early_stopping_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {early_stopping_patience} epochs without improvement")
                break
        
        total_time = time.time() - start_time
        print(f"Training completed in {total_time/60:.2f} minutes")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        print(f"Final checkpoint: checkpoint_epoch_{self.epoch+1}.pt")
        print(f"Best model: best_model.pt")
        
        self.writer.close()
    
    def evaluate_sample(self, structured_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate model on a single sample"""
        self.model.eval()
        
        with torch.no_grad():
            prediction = self.model([structured_inputs])
        
        return {
            'prediction': prediction[0].cpu().numpy(),
            'structured_inputs': structured_inputs
        }

def create_trainer(train_data_path: str,
                  val_data_path: str,
                  learning_rate: float = 1e-3,
                  batch_size: int = 32,
                  device: str = 'auto') -> SupervisedTrainer:
    """Create trainer with pre-generated datasets"""
    
    # Handle device detection
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load datasets
    train_dataset, val_dataset = load_train_val_datasets(train_data_path, val_data_path)
    
    # Create model
    from .transformer_model import create_model
    model = create_model(device=device)
    
    # Create trainer
    trainer = SupervisedTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        learning_rate=learning_rate,
        batch_size=batch_size
    )
    
    return trainer