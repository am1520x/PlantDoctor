"""
Data loading utilities with normalization for model training.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from typing import Tuple, List


def get_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: int = 224
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders with proper normalization for train/val/test sets.
    
    Args:
        data_dir: Path to processed dataset directory (should contain train/val/test folders)
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        img_size: Image size (should match preprocessing - default 224)
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    
    data_dir = Path(data_dir)
    
    # ImageNet normalization (standard for pre-trained models)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225]     # ImageNet std
    )
    
    # Training transforms (with additional augmentation)
    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),  # Converts to [0, 1] and changes to C, H, W format
        normalize
    ])
    
    # Validation/Test transforms (no augmentation, just normalization)
    val_test_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize
    ])
    
    # Create datasets using ImageFolder (expects structure: data_dir/class_name/images)
    train_dataset = datasets.ImageFolder(
        root=str(data_dir / "train"),
        transform=train_transforms
    )
    
    val_dataset = datasets.ImageFolder(
        root=str(data_dir / "val"),
        transform=val_test_transforms
    )
    
    test_dataset = datasets.ImageFolder(
        root=str(data_dir / "test"),
        transform=val_test_transforms
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data
        num_workers=num_workers,
        pin_memory=True  # Faster GPU transfer
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle test
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Dataset loaded from: {data_dir}")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Number of classes: {len(train_dataset.classes)}")
    print(f"  Batch size: {batch_size}")
    
    return train_loader, val_loader, test_loader


def get_class_names(data_dir: str) -> List[str]:
    """
    Get list of class names from the dataset.
    
    Args:
        data_dir: Path to processed dataset directory
        
    Returns:
        List of class names (sorted alphabetically)
    """
    data_dir = Path(data_dir)
    
    # Load from train directory to get class names
    train_dir = data_dir / "train"
    
    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    
    # Use ImageFolder to automatically detect classes
    dataset = datasets.ImageFolder(root=str(train_dir))
    
    # dataset.classes contains the folder names (class names)
    class_names = dataset.classes
    
    print(f"Found {len(class_names)} classes:")
    for i, name in enumerate(class_names[:10]):  # Show first 10
        print(f"  {i}: {name}")
    if len(class_names) > 10:
        print(f"  ... and {len(class_names) - 10} more")
    
    return class_names


def get_num_classes(data_dir: str) -> int:
    """
    Get the number of classes in the dataset.
    
    Args:
        data_dir: Path to processed dataset directory
        
    Returns:
        Number of classes
    """
    return len(get_class_names(data_dir))