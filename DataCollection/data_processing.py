"""
Data preprocessing and augmentation for plant image dataset.
Prepares images for training ML models.
"""

import os
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import json
import shutil
from PIL import Image
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Preprocess and augment plant images for ML training."""
    
    def __init__(
        self,
        source_dir: str,
        output_dir: str,
        img_size: Tuple[int, int] = (224, 224),
        validation_split: float = 0.2,
        test_split: float = 0.1
    ):
        """
        Initialize the image preprocessor.
        
        Args:
            source_dir: Directory containing raw images organized by class
            output_dir: Directory to save processed images
            img_size: Target image size (width, height)
            validation_split: Fraction of data for validation
            test_split: Fraction of data for testing
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.img_size = img_size
        self.validation_split = validation_split
        self.test_split = test_split
        
        # Create output directories
        self.train_dir = self.output_dir / "train"
        self.val_dir = self.output_dir / "val"
        self.test_dir = self.output_dir / "test"
        
        for dir_path in [self.train_dir, self.val_dir, self.test_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def get_dataset_statistics(self) -> Dict:
        """
        Analyze the dataset and return statistics.
        
        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            'total_classes': 0,
            'total_images': 0,
            'classes': {},
            'image_formats': set(),
            'corrupted_images': []
        }
        
        logger.info("Analyzing dataset...")
        
        # Iterate through class folders
        for class_dir in self.source_dir.iterdir():
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            image_count = 0
            
            # Count images in this class
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
                    try:
                        # Verify image can be opened
                        with Image.open(img_path) as img:
                            stats['image_formats'].add(img.format)
                        image_count += 1
                    except Exception as e:
                        logger.warning(f"Corrupted image: {img_path} - {e}")
                        stats['corrupted_images'].append(str(img_path))
            
            if image_count > 0:
                stats['classes'][class_name] = image_count
                stats['total_images'] += image_count
        
        stats['total_classes'] = len(stats['classes'])
        stats['image_formats'] = list(stats['image_formats'])
        
        return stats
    
    def preprocess_image(self, img_path: Path, output_path: Path) -> bool:
        """
        Preprocess a single image: resize, normalize, and save.
        
        Args:
            img_path: Path to input image
            output_path: Path to save processed image
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Open and convert to RGB
            with Image.open(img_path) as img:
                # Convert to RGB (handles grayscale and RGBA)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize with high-quality resampling
                img_resized = img.resize(self.img_size, Image.Resampling.LANCZOS)
                
                # Save as JPEG with good quality
                img_resized.save(output_path, 'JPEG', quality=95)
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")
            return False
    
    def split_and_preprocess(self, random_seed: int = 42) -> Dict:
        """
        Split dataset into train/val/test and preprocess all images.
        
        Args:
            random_seed: Random seed for reproducibility
            
        Returns:
            Dictionary with split statistics
        """
        import random
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        stats = {
            'train': {},
            'val': {},
            'test': {},
            'processed': 0,
            'failed': 0
        }
        
        logger.info("Splitting and preprocessing dataset...")
        
        # Process each class
        for class_dir in self.source_dir.iterdir():
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            logger.info(f"Processing class: {class_name}")
            
            # Create class directories in train/val/test
            (self.train_dir / class_name).mkdir(exist_ok=True)
            (self.val_dir / class_name).mkdir(exist_ok=True)
            (self.test_dir / class_name).mkdir(exist_ok=True)
            
            # Get all valid images
            image_files = [
                f for f in class_dir.iterdir()
                if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
            ]
            
            # Shuffle images
            random.shuffle(image_files)
            
            # Calculate split indices
            total = len(image_files)
            test_idx = int(total * self.test_split)
            val_idx = test_idx + int(total * self.validation_split)
            
            # Split files
            test_files = image_files[:test_idx]
            val_files = image_files[test_idx:val_idx]
            train_files = image_files[val_idx:]
            
            # Process and save images
            for split_name, files, split_dir in [
                ('train', train_files, self.train_dir),
                ('val', val_files, self.val_dir),
                ('test', test_files, self.test_dir)
            ]:
                count = 0
                for img_file in files:
                    output_path = split_dir / class_name / f"{img_file.stem}.jpg"
                    if self.preprocess_image(img_file, output_path):
                        count += 1
                        stats['processed'] += 1
                    else:
                        stats['failed'] += 1
                
                stats[split_name][class_name] = count
            
            logger.info(f"  Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
        
        return stats
    
    def create_augmented_images(self, augmentation_factor: int = 2) -> Dict:
        """
        Create augmented versions of training images to increase dataset size.
        
        Args:
            augmentation_factor: Number of augmented versions per image
            
        Returns:
            Dictionary with augmentation statistics
        """
        from PIL import ImageEnhance, ImageOps
        import random
        
        logger.info(f"Creating augmented images (factor: {augmentation_factor})...")
        
        stats = {
            'augmented': 0,
            'failed': 0
        }
        
        # Only augment training data
        for class_dir in self.train_dir.iterdir():
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            logger.info(f"Augmenting class: {class_name}")
            
            image_files = list(class_dir.glob("*.jpg"))
            
            for img_path in image_files:
                for aug_idx in range(augmentation_factor):
                    try:
                        with Image.open(img_path) as img:
                            # Apply random augmentations
                            augmented = img.copy()
                            
                            # Random horizontal flip
                            if random.random() > 0.5:
                                augmented = ImageOps.mirror(augmented)
                            
                            # Random rotation (-20 to 20 degrees)
                            angle = random.uniform(-20, 20)
                            augmented = augmented.rotate(angle, fillcolor=(255, 255, 255))
                            
                            # Random brightness adjustment
                            brightness = ImageEnhance.Brightness(augmented)
                            factor = random.uniform(0.8, 1.2)
                            augmented = brightness.enhance(factor)
                            
                            # Random contrast adjustment
                            contrast = ImageEnhance.Contrast(augmented)
                            factor = random.uniform(0.8, 1.2)
                            augmented = contrast.enhance(factor)
                            
                            # Random zoom (crop and resize)
                            if random.random() > 0.5:
                                width, height = augmented.size
                                crop_percent = random.uniform(0.8, 0.95)
                                crop_width = int(width * crop_percent)
                                crop_height = int(height * crop_percent)
                                left = random.randint(0, width - crop_width)
                                top = random.randint(0, height - crop_height)
                                augmented = augmented.crop((
                                    left, top,
                                    left + crop_width, top + crop_height
                                ))
                                augmented = augmented.resize(self.img_size, Image.Resampling.LANCZOS)
                            
                            # Save augmented image
                            aug_path = class_dir / f"{img_path.stem}_aug{aug_idx}.jpg"
                            augmented.save(aug_path, 'JPEG', quality=95)
                            stats['augmented'] += 1
                    
                    except Exception as e:
                        logger.error(f"Error augmenting {img_path}: {e}")
                        stats['failed'] += 1
            
            logger.info(f"  Created {augmentation_factor * len(image_files)} augmented images")
        
        return stats
    
    def save_metadata(self) -> None:
        """Save dataset metadata and statistics."""
        metadata = {
            'image_size': self.img_size,
            'splits': {
                'train': self.validation_split,
                'val': self.test_split,
                'test': 1.0 - self.validation_split - self.test_split
            },
            'source_directory': str(self.source_dir),
            'output_directory': str(self.output_dir)
        }
        
        # Count images per split
        for split_name, split_dir in [
            ('train', self.train_dir),
            ('val', self.val_dir),
            ('test', self.test_dir)
        ]:
            split_stats = {}
            total = 0
            for class_dir in split_dir.iterdir():
                if class_dir.is_dir():
                    count = len(list(class_dir.glob("*.jpg")))
                    split_stats[class_dir.name] = count
                    total += count
            metadata[split_name] = split_stats
            metadata[f'{split_name}_total'] = total
        
        # Save to JSON
        metadata_path = self.output_dir / "dataset_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata saved to: {metadata_path}")


def main():
    """Main preprocessing pipeline."""
    
    # Configure paths
    source_dir = r"d:\OneDrive\PlantDoctor\datasets\house_plant_species\house_plant_species"
    output_dir = r"d:\OneDrive\PlantDoctor\datasets\processed"
    
    # Initialize preprocessor
    preprocessor = ImagePreprocessor(
        source_dir=source_dir,
        output_dir=output_dir,
        img_size=(224, 224),  # Standard size for many CNN models
        validation_split=0.2,
        test_split=0.1
    )
    
    # Step 1: Analyze dataset
    logger.info("="*60)
    logger.info("STEP 1: ANALYZING DATASET")
    logger.info("="*60)
    stats = preprocessor.get_dataset_statistics()
    logger.info(f"Total classes: {stats['total_classes']}")
    logger.info(f"Total images: {stats['total_images']}")
    logger.info(f"Image formats: {stats['image_formats']}")
    logger.info(f"Corrupted images: {len(stats['corrupted_images'])}")
    
    if stats['corrupted_images']:
        logger.warning(f"Found {len(stats['corrupted_images'])} corrupted images")
    
    # Show class distribution
    logger.info("\nClass distribution:")
    for class_name, count in sorted(stats['classes'].items(), key=lambda x: x[1], reverse=True)[:10]:
        logger.info(f"  {class_name}: {count} images")
    
    # Step 2: Split and preprocess
    logger.info("\n" + "="*60)
    logger.info("STEP 2: SPLITTING AND PREPROCESSING")
    logger.info("="*60)
    split_stats = preprocessor.split_and_preprocess(random_seed=42)
    logger.info(f"Processed: {split_stats['processed']} images")
    logger.info(f"Failed: {split_stats['failed']} images")
    
    # Step 3: Data augmentation
    logger.info("\n" + "="*60)
    logger.info("STEP 3: DATA AUGMENTATION")
    logger.info("="*60)
    aug_stats = preprocessor.create_augmented_images(augmentation_factor=2)
    logger.info(f"Augmented: {aug_stats['augmented']} images")
    logger.info(f"Failed: {aug_stats['failed']} images")
    
    # Step 4: Save metadata
    logger.info("\n" + "="*60)
    logger.info("STEP 4: SAVING METADATA")
    logger.info("="*60)
    preprocessor.save_metadata()
    
    logger.info("\n" + "="*60)
    logger.info("PREPROCESSING COMPLETE!")
    logger.info("="*60)
    logger.info(f"Processed images saved to: {output_dir}")
    logger.info(f"  - Training set: {output_dir}/train")
    logger.info(f"  - Validation set: {output_dir}/val")
    logger.info(f"  - Test set: {output_dir}/test")


if __name__ == "__main__":
    main()