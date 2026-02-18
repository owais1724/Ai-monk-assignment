import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from sklearn.metrics import f1_score, accuracy_score
from typing import Tuple, List


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Get the best available device for computation.
    
    Returns:
        torch.device: Available device (cuda or cpu)
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device


def masked_bce_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, 
                   pos_weight: torch.Tensor = None) -> torch.Tensor:
    """
    Compute BCE loss with masking for NA labels.
    
    Args:
        logits (torch.Tensor): Model predictions (before sigmoid)
        targets (torch.Tensor): Ground truth labels
        mask (torch.Tensor): Binary mask (1 for valid, 0 for NA)
        pos_weight (torch.Tensor): Positive weights for class imbalance
    
    Returns:
        torch.Tensor: Computed loss
    """
    # Create BCEWithLogitsLoss with pos_weight if provided
    if pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
    else:
        criterion = nn.BCEWithLogitsLoss(reduction='none')
    
    # Compute loss
    loss = criterion(logits, targets)
    
    # Apply mask
    loss = loss * mask
    
    # Normalize by number of valid labels
    num_valid = mask.sum()
    if num_valid > 0:
        loss = loss.sum() / num_valid
    else:
        loss = torch.tensor(0.0, device=loss.device)
    
    return loss


def plot_loss_curve(losses: List[float], save_path: str = 'loss_curve.png'):
    """
    Plot and save training loss curve.
    
    Args:
        losses (List[float]): List of training losses
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(losses)), losses, 'b-', linewidth=2)
    plt.xlabel('iteration_number')
    plt.ylabel('training_loss')
    plt.title('Aimonk_multilabel_problem')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Loss curve saved to {save_path}")


def compute_metrics(predictions: torch.Tensor, targets: torch.Tensor, 
                   mask: torch.Tensor, threshold: float = 0.5) -> dict:
    """
    Compute multi-label classification metrics.
    
    Args:
        predictions (torch.Tensor): Model predictions (after sigmoid)
        targets (torch.Tensor): Ground truth labels
        mask (torch.Tensor): Binary mask (1 for valid, 0 for NA)
        threshold (float): Threshold for binary predictions
    
    Returns:
        dict: Dictionary containing metrics
    """
    # Convert to numpy and apply mask
    pred_np = predictions.detach().cpu().numpy()
    target_np = targets.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy()
    
    # Apply threshold to get binary predictions
    pred_binary = (pred_np >= threshold).astype(int)
    
    # Flatten arrays for metric computation
    pred_flat = pred_binary.flatten()
    target_flat = target_np.flatten()
    mask_flat = mask_np.flatten()
    
    # Apply mask to get only valid predictions
    valid_indices = mask_flat == 1
    pred_valid = pred_flat[valid_indices]
    target_valid = target_flat[valid_indices]
    
    if len(pred_valid) == 0:
        return {'accuracy': 0.0, 'f1_macro': 0.0, 'f1_micro': 0.0}
    
    # Compute metrics
    accuracy = accuracy_score(target_valid, pred_valid)
    f1_macro = f1_score(target_valid, pred_valid, average='macro', zero_division=0)
    f1_micro = f1_score(target_valid, pred_valid, average='micro', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro
    }


class EarlyStopping:
    """
    Early stopping utility to prevent overfitting.
    """
    
    def __init__(self, patience: int = 7, min_delta: float = 0.0, restore_best_weights: bool = True):
        """
        Args:
            patience (int): Number of epochs to wait before stopping
            min_delta (float): Minimum change to qualify as improvement
            restore_best_weights (bool): Whether to restore best model weights
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss (float): Current validation loss
            model (nn.Module): Current model
        
        Returns:
            bool: True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        
        return False


def create_directory_if_not_exists(directory: str):
    """
    Create directory if it doesn't exist.
    
    Args:
        directory (str): Directory path
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")


def print_model_info(model: nn.Module):
    """
    Print model information.
    
    Args:
        model (nn.Module): Model to analyze
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")


def get_learning_rate_scheduler(optimizer, scheduler_type: str = 'step', **kwargs):
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_type (str): Type of scheduler ('step', 'cosine', 'plateau')
        **kwargs: Additional arguments for scheduler
    
    Returns:
        Learning rate scheduler
    """
    if scheduler_type == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=kwargs.get('step_size', 10), 
            gamma=kwargs.get('gamma', 0.1)
        )
    elif scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=kwargs.get('T_max', 100)
        )
    elif scheduler_type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=kwargs.get('factor', 0.5), 
            patience=kwargs.get('patience', 5)
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
