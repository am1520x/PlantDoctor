"""
This code gets the plant image data to power the classification module from kaggle and huggingface.
"""

import os
# Set HuggingFace cache to D: drive to avoid space issues
os.environ['HF_HOME'] = r'd:\OneDrive\PlantDoctor\cache\huggingface'
os.environ['HF_DATASETS_CACHE'] = r'd:\OneDrive\PlantDoctor\cache\huggingface\datasets'
os.environ['HUGGINGFACE_HUB_CACHE'] = r'd:\OneDrive\PlantDoctor\cache\huggingface\hub'
os.environ['HF_HUB_CACHE'] = r'd:\OneDrive\PlantDoctor\cache\huggingface\hub'


from pathlib import Path
from typing import Optional, Dict, Any
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



class DatasetDownloader:
    """Base class for dataset downloading operations."""
    
    def __init__(self, base_dir: str = "datasets"):
        """
        Initialize the dataset downloader.
        
        Args:
            base_dir: Base directory to store downloaded datasets
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def get_dataset_path(self, dataset_name: str) -> Path:
        """Get the path for a specific dataset."""
        return self.base_dir / dataset_name


class DatasetValidator:
    """Validate and extract metadata from downloaded datasets."""
    
    @staticmethod
    def get_directory_size(path: Path) -> Dict[str, Any]:
        """
        Calculate total size of directory and its contents.
        
        Returns:
            Dictionary with size information in bytes, MB, and GB
        """
        total_size = 0
        for file_path in path.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        
        return {
            'bytes': total_size,
            'MB': round(total_size / (1024 * 1024), 2),
            'GB': round(total_size / (1024 * 1024 * 1024), 2)
        }
    
    @staticmethod
    def count_files_by_extension(path: Path) -> Dict[str, int]:
        """Count files grouped by extension."""
        extensions = {}
        for file_path in path.rglob('*'):
            if file_path.is_file():
                ext = file_path.suffix.lower() or 'no_extension'
                extensions[ext] = extensions.get(ext, 0) + 1
        return extensions
    
    @staticmethod
    def count_images(path: Path) -> int:
        """Count image files in directory."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
        count = 0
        for file_path in path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                count += 1
        return count
    
    @staticmethod
    def validate_kaggle_dataset(dataset_path: Path) -> Dict[str, Any]:
        """
        Validate and extract metadata from Kaggle dataset.
        
        Returns:
            Dictionary containing dataset metadata
        """
        metadata = {
            'path': str(dataset_path),
            'exists': dataset_path.exists(),
            'is_valid': False
        }
        
        if not dataset_path.exists():
            logger.warning(f"Dataset path does not exist: {dataset_path}")
            return metadata
        
        # Get size information
        metadata['size'] = DatasetValidator.get_directory_size(dataset_path)
        
        # Count files by extension
        metadata['file_counts'] = DatasetValidator.count_files_by_extension(dataset_path)
        
        # Count images
        metadata['total_images'] = DatasetValidator.count_images(dataset_path)
        
        # Check for CSV files (common in Kaggle datasets)
        csv_files = list(dataset_path.rglob('*.csv'))
        metadata['csv_files'] = [str(f.name) for f in csv_files]
        
        if csv_files:
            import pandas as pd
            csv_metadata = []
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    csv_metadata.append({
                        'filename': csv_file.name,
                        'rows': len(df),
                        'columns': list(df.columns),
                        'shape': df.shape,
                        'dtypes': df.dtypes.astype(str).to_dict()
                    })
                except Exception as e:
                    logger.error(f"Error reading CSV {csv_file}: {e}")
            metadata['csv_metadata'] = csv_metadata
        
        # List subdirectories (often represent classes/categories)
        subdirs = [d.name for d in dataset_path.iterdir() if d.is_dir()]
        metadata['subdirectories'] = subdirs
        metadata['num_subdirectories'] = len(subdirs)
        
        metadata['is_valid'] = metadata['total_images'] > 0 or len(csv_files) > 0
        
        return metadata
    
    @staticmethod
    def validate_huggingface_dataset(dataset_path: Path) -> Dict[str, Any]:
        """
        Validate and extract metadata from Hugging Face dataset.
        
        Returns:
            Dictionary containing dataset metadata
        """
        metadata = {
            'path': str(dataset_path),
            'exists': dataset_path.exists(),
            'is_valid': False
        }
        
        if not dataset_path.exists():
            logger.warning(f"Dataset path does not exist: {dataset_path}")
            return metadata
        
        try:
            from datasets import load_from_disk
            
            # Load dataset
            dataset = load_from_disk(str(dataset_path))
            
            # Extract metadata
            metadata['num_rows'] = len(dataset)
            metadata['num_columns'] = len(dataset.column_names)
            metadata['column_names'] = dataset.column_names
            metadata['features'] = str(dataset.features)
            
            # Get size information
            metadata['size'] = DatasetValidator.get_directory_size(dataset_path)
            
            # Check for image column
            image_columns = [col for col in dataset.column_names if 'image' in col.lower()]
            metadata['image_columns'] = image_columns
            
            if image_columns:
                metadata['total_images'] = len(dataset)
            
            # Get sample data (first row)
            if len(dataset) > 0:
                sample = dataset[0]
                metadata['sample_keys'] = list(sample.keys())
            
            # Check dataset splits if available
            if hasattr(dataset, 'keys'):
                metadata['splits'] = list(dataset.keys())
            
            metadata['is_valid'] = True
            
        except Exception as e:
            logger.error(f"Error validating Hugging Face dataset: {e}")
            metadata['error'] = str(e)
        
        return metadata


class KaggleDownloader(DatasetDownloader):
    """Download datasets from Kaggle."""
    
    def __init__(self, base_dir: str = "datasets"):
        super().__init__(base_dir)
        self._check_kaggle_credentials()
    
    def _check_kaggle_credentials(self):
        """Check if Kaggle API credentials are configured."""
        try:
            import kaggle
            logger.info("Kaggle credentials found")
        except OSError as e:
            logger.error("Kaggle credentials not found. Please set up ~/.kaggle/kaggle.json")
            raise
    
    def download(self, dataset_id: str, dataset_name: Optional[str] = None, unzip: bool = True) -> Path:
        """
        Download a dataset from Kaggle.
        
        Args:
            dataset_id: Kaggle dataset ID (e.g., 'prakash27x/indoor-house-plants-dataset-with-care-instructions')
            dataset_name: Optional custom name for the dataset folder
            unzip: Whether to unzip the downloaded files
            
        Returns:
            Path to the downloaded dataset
        """
        import kaggle
        
        if dataset_name is None:
            dataset_name = dataset_id.split('/')[-1]
        
        dataset_path = self.get_dataset_path(dataset_name)
        dataset_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Downloading Kaggle dataset: {dataset_id}")
        kaggle.api.dataset_download_files(
            dataset_id,
            path=str(dataset_path),
            unzip=unzip,
            quiet=False
        )
        logger.info(f"Dataset downloaded to: {dataset_path}")
        return dataset_path
    
    def validate(self, dataset_path: Path) -> Dict[str, Any]:
        """Validate Kaggle dataset."""
        return DatasetValidator.validate_kaggle_dataset(dataset_path)


class HuggingFaceDownloader(DatasetDownloader):
    """Download datasets from Hugging Face."""
    
    def download(self, dataset_id: str, dataset_name: Optional[str] = None, split: Optional[str] = None) -> Path:
        """
        Download a dataset from Hugging Face.
        
        Args:
            dataset_id: Hugging Face dataset ID (e.g., 'kakasher/house-plant-species')
            dataset_name: Optional custom name for the dataset folder
            split: Optional dataset split to download (e.g., 'train', 'test')
            
        Returns:
            Path to the downloaded dataset
        """
        from huggingface_hub import snapshot_download
        
        if dataset_name is None:
            dataset_name = dataset_id.split('/')[-1]
        
        dataset_path = self.get_dataset_path(dataset_name)
        dataset_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Downloading Hugging Face dataset: {dataset_id}")
        
        try:
            # Download entire repository instead of using load_dataset
            logger.info("Downloading dataset files directly...")
            downloaded_path = snapshot_download(
                repo_id=dataset_id,
                repo_type="dataset",
                local_dir=str(dataset_path),
                local_dir_use_symlinks=False,
                resume_download=True
            )
            
            logger.info(f"Dataset downloaded to: {dataset_path}")
            
            # Try to extract the tar file if it exists
            tar_files = list(dataset_path.glob("*.tar"))
            if tar_files:
                import tarfile
                logger.info(f"Extracting tar file: {tar_files[0].name}")
                with tarfile.open(tar_files[0], 'r') as tar:
                    tar.extractall(path=dataset_path / "extracted")
                logger.info("Extraction complete")
            
            return dataset_path
            
        except Exception as e:
            logger.error(f"Download failed: {e}")
            # Try streaming as fallback
            logger.info("Attempting streaming download as fallback...")
            from datasets import load_dataset
            dataset = load_dataset(dataset_id, split=split, streaming=True)
            
            # Save streaming data in batches
            batch_size = 100
            batch_data = []
            output_path = dataset_path / "data"
            output_path.mkdir(exist_ok=True)
            
            batch_num = 0
            for idx, item in enumerate(dataset):
                batch_data.append(item)
                if len(batch_data) >= batch_size:
                    # Save batch
                    import pickle
                    with open(output_path / f"batch_{batch_num}.pkl", 'wb') as f:
                        pickle.dump(batch_data, f)
                    logger.info(f"Saved batch {batch_num} ({len(batch_data)} items)")
                    batch_data = []
                    batch_num += 1
            
            # Save remaining data
            if batch_data:
                import pickle
                with open(output_path / f"batch_{batch_num}.pkl", 'wb') as f:
                    pickle.dump(batch_data, f)
                logger.info(f"Saved final batch {batch_num} ({len(batch_data)} items)")
            
            logger.info(f"Streaming download complete: {batch_num + 1} batches saved")
            return dataset_path
    
    def validate(self, dataset_path: Path) -> Dict[str, Any]:
        """Validate Hugging Face dataset."""
        return DatasetValidator.validate_huggingface_dataset(dataset_path)
    

class PlantDatasetManager:
    """Manage plant image datasets from multiple sources."""
    
    def __init__(self, base_dir: str = "datasets"):
        """Initialize the plant dataset manager."""
        self.kaggle_downloader = KaggleDownloader(base_dir)
        self.hf_downloader = HuggingFaceDownloader(base_dir)
        self.metadata = {}
    
    def download_indoor_plants_kaggle(self) -> Path:
        """Download the indoor house plants dataset from Kaggle."""
        return self.kaggle_downloader.download(
            dataset_id="prakash27x/indoor-house-plants-dataset-with-care-instructions",
            dataset_name="indoor_house_plants"
        )
    
    def download_house_plant_species_hf(self, split: Optional[str] = None) -> Path:
        """Download the house plant species dataset from Hugging Face."""
        return self.hf_downloader.download(
            dataset_id="kakasher/house-plant-species",
            dataset_name="house_plant_species",
            split=split
        )
    
    def download_all(self) -> Dict[str, Path]:
        """
        Download all configured plant datasets.
        
        Returns:
            Dictionary with dataset names and their paths
        """
        datasets = {}
        
        try:
            datasets['indoor_plants'] = self.download_indoor_plants_kaggle()
        except Exception as e:
            logger.error(f"Failed to download Kaggle dataset: {e}")
        
        try:
            datasets['house_plant_species'] = self.download_house_plant_species_hf()
        except Exception as e:
            logger.error(f"Failed to download Hugging Face dataset: {e}")
        
        return datasets
    
    def validate_all_datasets(self) -> Dict[str, Dict[str, Any]]:
        """
        Validate all downloaded datasets and extract metadata.
        
        Returns:
            Dictionary with dataset names and their metadata
        """
        validation_results = {}
        
        # Validate Kaggle dataset
        kaggle_path = self.kaggle_downloader.get_dataset_path("indoor_house_plants")
        if kaggle_path.exists():
            logger.info("\n" + "="*60)
            logger.info("Validating Kaggle Dataset: Indoor House Plants")
            logger.info("="*60)
            validation_results['indoor_plants'] = self.kaggle_downloader.validate(kaggle_path)
            self._print_metadata(validation_results['indoor_plants'])
        
        # Validate Hugging Face dataset
        hf_path = self.hf_downloader.get_dataset_path("house_plant_species")
        if hf_path.exists():
            logger.info("\n" + "="*60)
            logger.info("Validating Hugging Face Dataset: House Plant Species")
            logger.info("="*60)
            validation_results['house_plant_species'] = self.hf_downloader.validate(hf_path)
            self._print_metadata(validation_results['house_plant_species'])
        
        self.metadata = validation_results
        return validation_results
    
    def _print_metadata(self, metadata: Dict[str, Any]):
        """Pretty print metadata."""
        logger.info(f"Path: {metadata['path']}")
        logger.info(f"Exists: {metadata['exists']}")
        logger.info(f"Valid: {metadata['is_valid']}")
        
        if 'size' in metadata:
            logger.info(f"Size: {metadata['size']['MB']} MB ({metadata['size']['GB']} GB)")
        
        if 'total_images' in metadata:
            logger.info(f"Total Images: {metadata['total_images']}")
        
        if 'num_rows' in metadata:
            logger.info(f"Number of Rows: {metadata['num_rows']}")
        
        if 'column_names' in metadata:
            logger.info(f"Columns: {metadata['column_names']}")
        
        if 'csv_metadata' in metadata:
            logger.info("CSV Files:")
            for csv in metadata['csv_metadata']:
                logger.info(f"  - {csv['filename']}: {csv['rows']} rows, {len(csv['columns'])} columns")
                logger.info(f"    Columns: {csv['columns']}")
        
        if 'file_counts' in metadata:
            logger.info(f"File Types: {metadata['file_counts']}")
        
        if 'subdirectories' in metadata and metadata['subdirectories']:
            logger.info(f"Subdirectories ({metadata['num_subdirectories']}): {metadata['subdirectories'][:10]}...")
    
    def save_metadata(self, output_file: str = "dataset_metadata.json"):
        """Save metadata to JSON file."""
        output_path = Path(self.kaggle_downloader.base_dir) / output_file
        with open(output_path, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
        logger.info(f"Metadata saved to: {output_path}")


def main():
    """Main function to demonstrate dataset downloading and validation."""
    manager = PlantDatasetManager(base_dir="datasets")
    
    # Download all datasets
    logger.info("Starting dataset downloads...")
    downloaded_datasets = manager.download_all()
    
    logger.info("\nDownload Summary:")
    for name, path in downloaded_datasets.items():
        logger.info(f"  {name}: {path}")
    
    # Validate and extract metadata
    logger.info("\n" + "="*60)
    logger.info("VALIDATING DATASETS")
    logger.info("="*60)
    manager.validate_all_datasets()
    
    # Save metadata to file
    manager.save_metadata()


if __name__ == "__main__":
    main()