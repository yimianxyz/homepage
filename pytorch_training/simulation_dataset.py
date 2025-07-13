"""
Simulation Dataset - Load pre-generated supervised learning data

Simple dataset that loads pre-generated training data from disk.
"""

import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Tuple

class SimulationDataset(Dataset):
    """Simple dataset that loads pre-generated training data"""
    
    def __init__(self, data_path: str):
        """
        Args:
            data_path: Path to pickled data file
        """
        self.data_path = data_path
        
        # Load data
        print(f"Loading data from {data_path}...")
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        
        print(f"Loaded {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, Any], torch.Tensor]:
        """Get a training sample"""
        sample = self.data[idx]
        
        # Convert target to tensor
        target = torch.tensor(sample['target'], dtype=torch.float32)
        
        return sample['inputs'], target

def collate_fn(batch: List[Tuple[Dict[str, Any], torch.Tensor]]) -> Tuple[List[Dict[str, Any]], torch.Tensor]:
    """
    Custom collate function to handle structured inputs
    
    Args:
        batch: List of (structured_inputs, target) tuples
        
    Returns:
        Tuple of (batch_structured_inputs, batch_targets)
    """
    # Separate structured inputs and targets
    structured_inputs, targets = zip(*batch)
    
    # Stack targets into tensor
    targets = torch.stack(targets)
    
    # Keep structured inputs as list (will be processed by model)
    return list(structured_inputs), targets

def create_dataloader(dataset: SimulationDataset, 
                     batch_size: int = 32, 
                     shuffle: bool = True) -> DataLoader:
    """Create DataLoader for SimulationDataset"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )

def load_train_val_datasets(train_data_path: str, 
                           val_data_path: str) -> Tuple[SimulationDataset, SimulationDataset]:
    """Load training and validation datasets from disk"""
    
    train_dataset = SimulationDataset(train_data_path)
    val_dataset = SimulationDataset(val_data_path)
    
    return train_dataset, val_dataset

if __name__ == "__main__":
    # Test dataset creation
    print("Testing SimulationDataset...")
    
    # Create small test dataset
    dataset = SimulationDataset(data_path="test_data.pkl")
    
    print(f"Dataset length: {len(dataset)}")
    
    # Test single sample
    sample_inputs, sample_target = dataset[0]
    print(f"Sample structured inputs keys: {sample_inputs.keys()}")
    print(f"Sample target shape: {sample_target.shape}")
    print(f"Sample target: {sample_target}")
    
    # Test dataloader
    dataloader = create_dataloader(dataset, batch_size=4)
    
    for batch_idx, (batch_inputs, batch_targets) in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print(f"  Batch size: {len(batch_inputs)}")
        print(f"  Target shape: {batch_targets.shape}")
        print(f"  First sample boids: {len(batch_inputs[0]['boids'])}")
        if batch_idx >= 2:  # Only test a few batches
            break
    
    print("Dataset testing complete!") 