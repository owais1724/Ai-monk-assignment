import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
import numpy as np
from typing import Tuple, Optional


class MultiLabelDataset(Dataset):
    """
    Multi-label image dataset with NA label handling.
    """
    
    def __init__(self, images_dir: str, labels_file: str, transform=None):
        """
        Args:
            images_dir (str): Directory with all the images
            labels_file (str): Path to the labels file
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.images_dir = images_dir
        self.transform = transform
        
        # Read labels file
        self.labels_df = pd.read_csv(labels_file, sep='\t', header=None, na_filter=False)
        self.labels_df.columns = ['image_name'] + [f'attr_{i+1}' for i in range(4)]
        
        # Get image paths and labels
        self.image_paths = []
        self.labels = []
        self.masks = []  # Mask for NA handling (1 for valid, 0 for NA)
        
        for idx, row in self.labels_df.iterrows():
            if idx == 0:  # Skip header row
                continue
                
            img_path = os.path.join(images_dir, row['image_name'])
            if os.path.exists(img_path):
                self.image_paths.append(img_path)
                
                # Process labels
                label_list = []
                mask_list = []
                
                for i in range(4):
                    attr_val = row[f'attr_{i+1}']
                    if attr_val == 'NA':
                        label_list.append(0.0)  # Replace NA with 0
                        mask_list.append(0.0)   # Mark as NA
                    else:
                        label_list.append(float(attr_val))
                        mask_list.append(1.0)   # Mark as valid
                
                self.labels.append(label_list)
                self.masks.append(mask_list)
        
        self.labels = torch.tensor(self.labels, dtype=torch.float32)
        self.masks = torch.tensor(self.masks, dtype=torch.float32)
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            image: Transformed image tensor
            labels: Multi-label tensor (with NA replaced by 0)
            mask: Binary mask for NA handling (1 for valid, 0 for NA)
        """
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get labels and mask
        labels = self.labels[idx]
        mask = self.masks[idx]
        
        return image, labels, mask


def get_data_loaders(images_dir: str, labels_file: str, batch_size: int = 32, 
                    num_workers: int = 4, train_split: float = 0.8) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders.
    
    Args:
        images_dir (str): Directory with all the images
        labels_file (str): Path to the labels file
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of worker processes for data loading
        train_split (float): Fraction of data to use for training
    
    Returns:
        Tuple[DataLoader, DataLoader]: Train and validation data loaders
    """
    
    # Data augmentation and normalization
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    full_dataset = MultiLabelDataset(images_dir, labels_file, transform=None)
    
    # Split dataset
    dataset_size = len(full_dataset)
    train_size = int(train_split * dataset_size)
    val_size = dataset_size - train_size
    
    train_indices, val_indices = torch.utils.data.random_split(
        range(dataset_size), [train_size, val_size]
    )
    
    # Create subsets with appropriate transforms
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    
    # Apply transforms to subsets
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def compute_class_weights(labels_file: str) -> torch.Tensor:
    """
    Compute class weights for handling imbalanced dataset.
    
    Args:
        labels_file (str): Path to the labels file
    
    Returns:
        torch.Tensor: Class weights for BCEWithLogitsLoss
    """
    # Read labels
    labels_df = pd.read_csv(labels_file, sep='\t', header=None, na_filter=False)
    labels_df.columns = ['image_name'] + [f'attr_{i+1}' for i in range(4)]
    
    # Count positive and negative samples for each class
    pos_counts = np.zeros(4)
    neg_counts = np.zeros(4)
    
    for idx, row in labels_df.iterrows():
        if idx == 0:  # Skip header
            continue
            
        for i in range(4):
            attr_val = row[f'attr_{i+1}']
            if attr_val != 'NA':
                if attr_val == '1':
                    pos_counts[i] += 1
                else:
                    neg_counts[i] += 1
    
    # Compute weights (negative_count / positive_count)
    weights = []
    for i in range(4):
        if pos_counts[i] > 0:
            weight = neg_counts[i] / pos_counts[i]
        else:
            weight = 1.0
        weights.append(weight)
    
    return torch.tensor(weights, dtype=torch.float32)
