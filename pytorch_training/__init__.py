"""
PyTorch Training Module

Provides supervised learning training for the transformer predator.
"""

from .transformer_model import TransformerPredator, create_model
from .teacher_policy import TeacherPolicy
from .simulation_dataset import SimulationDataset, load_train_val_datasets, create_dataloader
from .supervised_trainer import SupervisedTrainer, create_trainer

__all__ = [
    'TransformerPredator',
    'create_model',
    'TeacherPolicy', 
    'SimulationDataset',
    'load_train_val_datasets',
    'create_dataloader',
    'SupervisedTrainer',
    'create_trainer'
]

__version__ = "1.0.0" 