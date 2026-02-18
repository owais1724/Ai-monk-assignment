import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import os
import time
from tqdm import tqdm

from dataset import get_data_loaders, compute_class_weights
from model import get_model, save_model
from utils import set_seed, get_device, masked_bce_loss, plot_loss_curve, EarlyStopping, print_model_info, get_learning_rate_scheduler


# ==================== CONFIGURATION ====================
CONFIG = {
    # Data settings
    'images_dir': 'images',
    'labels_file': 'labels.txt',
    'batch_size': 4,
    'num_workers': 0,
    'train_split': 0.8,
    
    # Model settings
    'model_name': 'resnet50',  # 'resnet50' or 'mobilenetv2'
    'num_classes': 4,
    'pretrained': True,
    
    # Training settings
    'epochs': 5,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'seed': 42,
    
    # Loss settings
    'use_class_weights': True,  # Handle imbalance
    'use_focal_loss': False,    # Alternative to class weights
    
    # Scheduler settings
    'use_scheduler': True,
    'scheduler_type': 'step',  # 'step', 'cosine', 'plateau'
    'step_size': 2,
    'gamma': 0.1,
    
    # Early stopping
    'use_early_stopping': True,
    'patience': 3,
    'min_delta': 0.001,
    
    # Mixed precision
    'use_mixed_precision': False,
    
    # Model saving
    'model_save_path': 'multilabel_model.pth',
    'save_best_only': True,
    
    # Logging
    'print_freq': 1,
    'plot_loss_curve': True,
    'loss_curve_path': 'loss_curve.png'
}


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'none'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def train_epoch(model, train_loader, criterion, optimizer, device, scaler=None, use_focal_loss=False):
    """
    Train the model for one epoch.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use
        scaler: GradScaler for mixed precision
        use_focal_loss: Whether to use focal loss
    
    Returns:
        float: Average training loss
    """
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)
    
    progress_bar = tqdm(train_loader, desc='Training', leave=False)
    
    for batch_idx, (images, labels, mask) in enumerate(progress_bar):
        images = images.to(device)
        labels = labels.to(device)
        mask = mask.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        if scaler is not None:
            with autocast():
                outputs = model(images)
                
                if use_focal_loss:
                    loss = criterion(outputs, labels)
                    loss = (loss * mask).sum() / mask.sum()
                else:
                    loss = criterion(outputs, labels, mask)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            
            if use_focal_loss:
                loss = criterion(outputs, labels)
                loss = (loss * mask).sum() / mask.sum()
            else:
                loss = criterion(outputs, labels, mask)
            
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({'loss': loss.item()})
        
        if batch_idx % CONFIG['print_freq'] == 0:
            print(f'Batch [{batch_idx}/{num_batches}], Loss: {loss.item():.6f}')
    
    return total_loss / num_batches


def validate_epoch(model, val_loader, criterion, device, use_focal_loss=False):
    """
    Validate the model for one epoch.
    
    Args:
        model: PyTorch model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to use
        use_focal_loss: Whether to use focal loss
    
    Returns:
        float: Average validation loss
    """
    model.eval()
    total_loss = 0.0
    num_batches = len(val_loader)
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc='Validation', leave=False)
        
        for images, labels, mask in progress_bar:
            images = images.to(device)
            labels = labels.to(device)
            mask = mask.to(device)
            
            outputs = model(images)
            
            if use_focal_loss:
                loss = criterion(outputs, labels)
                loss = (loss * mask).sum() / mask.sum()
            else:
                loss = criterion(outputs, labels, mask)
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / num_batches


def main():
    """
    Main training function.
    """
    # Set random seed for reproducibility
    set_seed(CONFIG['seed'])
    
    # Get device
    device = get_device()
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = get_data_loaders(
        images_dir=CONFIG['images_dir'],
        labels_file=CONFIG['labels_file'],
        batch_size=CONFIG['batch_size'],
        num_workers=CONFIG['num_workers'],
        train_split=CONFIG['train_split']
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    print("Creating model...")
    model = get_model(
        model_name=CONFIG['model_name'],
        num_classes=CONFIG['num_classes'],
        pretrained=CONFIG['pretrained']
    )
    model = model.to(device)
    
    # Print model info
    print_model_info(model)
    
    # Handle class imbalance
    pos_weight = None
    if CONFIG['use_class_weights'] and not CONFIG['use_focal_loss']:
        print("Computing class weights for imbalance handling...")
        pos_weight = compute_class_weights(CONFIG['labels_file'])
        pos_weight = pos_weight.to(device)
        print(f"Class weights: {pos_weight}")
    
    # Create loss function
    if CONFIG['use_focal_loss']:
        criterion = FocalLoss(alpha=1.0, gamma=2.0, reduction='none')
        print("Using Focal Loss for imbalance handling")
    else:
        # We'll use masked_bce_loss which handles NA labels
        criterion = lambda logits, targets, mask: masked_bce_loss(logits, targets, mask, pos_weight)
        print("Using BCEWithLogitsLoss with class weights")
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    
    # Create learning rate scheduler
    scheduler = None
    if CONFIG['use_scheduler']:
        scheduler = get_learning_rate_scheduler(
            optimizer,
            scheduler_type=CONFIG['scheduler_type'],
            step_size=CONFIG['step_size'],
            gamma=CONFIG['gamma']
        )
        print(f"Using {CONFIG['scheduler_type']} scheduler")
    
    # Create early stopping
    early_stopping = None
    if CONFIG['use_early_stopping']:
        early_stopping = EarlyStopping(
            patience=CONFIG['patience'],
            min_delta=CONFIG['min_delta'],
            restore_best_weights=True
        )
        print(f"Using early stopping with patience={CONFIG['patience']}")
    
    # Create gradient scaler for mixed precision
    scaler = None
    if CONFIG['use_mixed_precision']:
        scaler = GradScaler()
        print("Using mixed precision training")
    
    # Training loop
    print("Starting training...")
    train_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(CONFIG['epochs']):
        print(f"\nEpoch [{epoch+1}/{CONFIG['epochs']}]")
        print("-" * 50)
        
        model.train()
        progress_bar = tqdm(train_loader, desc='Training', leave=False)
        
        for batch_idx, (images, labels, mask) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)
            mask = mask.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels, mask)
            
            loss.backward()
            optimizer.step()
            
            # Record per-iteration loss
            train_losses.append(loss.item())
            
            progress_bar.set_postfix({'loss': loss.item()})
            if batch_idx % CONFIG['print_freq'] == 0:
                print(f'Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.6f}')
        
        # Validate
        val_loss = validate_epoch(
            model, val_loader, criterion, device, CONFIG['use_focal_loss']
        )
        
        # Update learning rate
        if scheduler is not None:
            if CONFIG['scheduler_type'] == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
                
        print(f"Epoch [{epoch+1}/{CONFIG['epochs']}] Val Loss: {val_loss:.6f}")
        
        if scheduler is not None:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if CONFIG['save_best_only']:
                save_model(
                    model, CONFIG['model_save_path'], optimizer, epoch+1, best_val_loss
                )
                print(f"New best model saved with val_loss: {best_val_loss:.6f}")
        
        # Early stopping check
        if early_stopping is not None:
            if early_stopping(val_loss, model):
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
    
    # Save final model if not using best_only
    if not CONFIG['save_best_only']:
        save_model(model, CONFIG['model_save_path'], optimizer, CONFIG['epochs'], best_val_loss)
    
    # Plot loss curve
    if CONFIG['plot_loss_curve']:
        plot_loss_curve(train_losses, CONFIG['loss_curve_path'])
    
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved to: {CONFIG['model_save_path']}")


if __name__ == '__main__':
    main()
