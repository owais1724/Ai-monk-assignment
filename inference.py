import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import argparse
import os
import sys

from model import get_model, load_model
from utils import get_device


class MultiLabelInference:
    """
    Multi-label image classification inference class.
    """
    
    def __init__(self, model_path: str, model_name: str = 'resnet50', num_classes: int = 4):
        """
        Initialize inference class.
        
        Args:
            model_path (str): Path to the trained model
            model_name (str): Model architecture name
            num_classes (int): Number of output classes
        """
        self.device = get_device()
        self.model_name = model_name
        self.num_classes = num_classes
        self.threshold = 0.5
        
        # Load model
        self.model = get_model(model_name=model_name, num_classes=num_classes, pretrained=False)
        self.model = self.model.to(self.device)
        
        # Load trained weights
        if os.path.exists(model_path):
            load_model(model_path, self.model)
            print(f"Model loaded from {model_path}")
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Define preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Attribute names
        self.attribute_names = [f'Attr{i+1}' for i in range(num_classes)]
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Preprocess input image.
        
        Args:
            image_path (str): Path to the input image
        
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
        return image_tensor.to(self.device)
    
    def predict(self, image_path: str) -> dict:
        """
        Perform inference on a single image.
        
        Args:
            image_path (str): Path to the input image
        
        Returns:
            dict: Prediction results containing probabilities and binary predictions
        """
        # Preprocess image
        image_tensor = self.preprocess_image(image_path)
        
        # Perform inference
        with torch.no_grad():
            logits = self.model(image_tensor)
            probabilities = torch.sigmoid(logits)
            binary_predictions = (probabilities >= self.threshold).float()
        
        # Convert to numpy for easier handling
        probs = probabilities.cpu().numpy().flatten()
        binary_preds = binary_predictions.cpu().numpy().flatten()
        
        # Prepare results
        results = {
            'image_path': image_path,
            'probabilities': {},
            'binary_predictions': {},
            'present_attributes': []
        }
        
        # Fill results
        for i, attr_name in enumerate(self.attribute_names):
            results['probabilities'][attr_name] = float(probs[i])
            results['binary_predictions'][attr_name] = int(binary_preds[i])
            
            if binary_preds[i] == 1:
                results['present_attributes'].append(attr_name)
        
        return results
    
    def print_results(self, results: dict):
        """
        Print prediction results in a formatted way.
        
        Args:
            results (dict): Prediction results from predict() method
        """
        print(f"\n{'='*60}")
        print(f"INFERENCE RESULTS FOR: {os.path.basename(results['image_path'])}")
        print(f"{'='*60}")
        
        print(f"\nProbabilities:")
        for attr_name, prob in results['probabilities'].items():
            print(f"  {attr_name}: {prob:.4f}")
        
        print(f"\nBinary Predictions (threshold={self.threshold}):")
        for attr_name, pred in results['binary_predictions'].items():
            status = "Present" if pred == 1 else "Absent"
            print(f"  {attr_name}: {status}")
        
        print(f"\nAttributes present: {results['present_attributes']}")
        print(f"{'='*60}\n")


def main():
    """
    Main inference function.
    """
    parser = argparse.ArgumentParser(description='Multi-label Image Classification Inference')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to the input image')
    parser.add_argument('--model_path', type=str, default='multilabel_model.pth',
                        help='Path to the trained model')
    parser.add_argument('--model_name', type=str, default='resnet50',
                        choices=['resnet50', 'mobilenetv2'],
                        help='Model architecture')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for binary classification')
    parser.add_argument('--num_classes', type=int, default=4,
                        help='Number of output classes')
    
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image not found at {args.image_path}")
        sys.exit(1)
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        sys.exit(1)
    
    try:
        # Initialize inference class
        inference = MultiLabelInference(
            model_path=args.model_path,
            model_name=args.model_name,
            num_classes=args.num_classes
        )
        
        # Update threshold if provided
        inference.threshold = args.threshold
        
        # Perform inference
        print(f"Performing inference on: {args.image_path}")
        print(f"Using model: {args.model_name}")
        print(f"Threshold: {args.threshold}")
        
        results = inference.predict(args.image_path)
        
        # Print results
        inference.print_results(results)
        
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        sys.exit(1)


def batch_inference(image_dir: str, model_path: str, model_name: str = 'resnet50', 
                   threshold: float = 0.5, output_file: str = None):
    """
    Perform batch inference on all images in a directory.
    
    Args:
        image_dir (str): Directory containing images
        model_path (str): Path to the trained model
        model_name (str): Model architecture
        threshold (float): Threshold for binary classification
        output_file (str): Optional output file to save results
    """
    # Initialize inference class
    inference = MultiLabelInference(
        model_path=model_path,
        model_name=model_name,
        num_classes=4
    )
    inference.threshold = threshold
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = []
    
    for file in os.listdir(image_dir):
        if os.path.splitext(file.lower())[1] in image_extensions:
            image_files.append(os.path.join(image_dir, file))
    
    if not image_files:
        print(f"No image files found in {image_dir}")
        return
    
    print(f"Found {len(image_files)} images. Starting batch inference...")
    
    # Store results
    all_results = []
    
    for image_path in image_files:
        try:
            results = inference.predict(image_path)
            all_results.append(results)
            
            # Print results
            inference.print_results(results)
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
    
    # Save results to file if specified
    if output_file:
        import json
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {output_file}")


if __name__ == '__main__':
    # Check if running in batch mode or single image mode
    if len(sys.argv) > 1 and sys.argv[1] == '--batch':
        # Batch mode
        parser = argparse.ArgumentParser(description='Batch Multi-label Image Classification Inference')
        parser.add_argument('--image_dir', type=str, required=True,
                            help='Directory containing images')
        parser.add_argument('--model_path', type=str, default='multilabel_model.pth',
                            help='Path to the trained model')
        parser.add_argument('--model_name', type=str, default='resnet50',
                            choices=['resnet50', 'mobilenetv2'],
                            help='Model architecture')
        parser.add_argument('--threshold', type=float, default=0.5,
                            help='Threshold for binary classification')
        parser.add_argument('--output_file', type=str, default=None,
                            help='Output file to save results (JSON format)')
        
        args = parser.parse_args(sys.argv[2:])
        
        batch_inference(
            image_dir=args.image_dir,
            model_path=args.model_path,
            model_name=args.model_name,
            threshold=args.threshold,
            output_file=args.output_file
        )
    else:
        # Single image mode
        main()
