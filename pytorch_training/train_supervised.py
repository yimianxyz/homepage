#!/usr/bin/env python3
"""
Train Supervised Learning Model

Simple training script that uses pre-generated data.
"""

import argparse
import os
from pathlib import Path

from .supervised_trainer import create_trainer

def main():
    parser = argparse.ArgumentParser(description='Train supervised learning model')
    parser.add_argument('--train-data', type=str, required=True, help='Path to training data file')
    parser.add_argument('--val-data', type=str, required=True, help='Path to validation data file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--device', type=str, default='auto', help='Device (cuda/cpu/auto)')
    parser.add_argument('--early-stopping', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Check data files exist
    if not os.path.exists(args.train_data):
        print(f"Error: Training data file not found: {args.train_data}")
        print("Generate data first using: python generate_data.py")
        return
    
    if not os.path.exists(args.val_data):
        print(f"Error: Validation data file not found: {args.val_data}")
        print("Generate data first using: python generate_data.py")
        return
    
    print("=== Supervised Learning Training ===")
    print(f"Training data: {args.train_data}")
    print(f"Validation data: {args.val_data}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Device: {args.device}")
    print()
    
    # Create trainer
    trainer = create_trainer(
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        device=args.device
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
        print(f"Resumed from checkpoint: {args.resume}")
    
    # Train model
        trainer.train(
            num_epochs=args.epochs,
            early_stopping_patience=args.early_stopping
        )
        
    print("Training complete!")

if __name__ == "__main__":
    main() 