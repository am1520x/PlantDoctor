import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from pathlib import Path

from Models.models import ModelFactory
from Models.data_loader import get_class_names


def load_inference_model(model_name, num_classes, model_path, device='cpu'):
    """
    Instantiate architecture and load trained weights.
    """
    model = ModelFactory.create_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False
    )

    checkpoint = torch.load(model_path, map_location=device)
    
    # Check if checkpoint is a dict with 'model_state_dict' key
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()  # Set to evaluation mode (important for BatchNorm/Dropout)
    return model

def predict_image(model, image_path, class_names, device='cpu'):
    """
    Preprocess the image and return predicted class and confidence.
    """
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(image_path).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted_idx = torch.max(probabilities, dim=1)

    return {
        "class" : class_names[predicted_idx.item()],
        "confidence" : confidence.item()
    }

if __name__ == "__main__":
    DATA_DIR = "./datasets/processed"
    MODEL_FILE = "experiments/efficientnet_b0/best_model.pth"
    MODEL_NAME = "efficientnet_b0" # Example - must match training choice [cite: 32]
    
    # Get class names from your existing data_loader utility 
    classes = get_class_names(DATA_DIR)
    
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_inference_model(MODEL_NAME, len(classes), MODEL_FILE, device)
    
    # Run prediction
    result = predict_image(model, "datasets/test_plant_2.jpg", classes, device)
    print(f"Prediction: {result['class']} ({result['confidence']:.2%})")