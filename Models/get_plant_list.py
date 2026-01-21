import os
import sys

# Add the Models directory to the path to import data_loader
sys.path.append(os.path.dirname(__file__))

from data_loader import get_class_names

def generate_class_list(dataset_path='./datasets/processed', output_file='./API/plant_classes.txt'):
    """Generate a text file containing the list of plant classes."""
    # Reuse the existing get_class_names function
    classes = get_class_names(dataset_path)
    
    with open(output_file, 'w') as f:
        for cls in classes:
            f.write(f"{cls}\n")
    
    print(f"Generated {output_file} with {len(classes)} classes")
    return classes

if __name__ == "__main__":
    generate_class_list()