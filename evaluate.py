import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from data.dataset import create_data_loaders
from models.models import get_model
from utils.train_utils import test_model, plot_confusion_matrix, plot_roc_curve
from utils.visualization import visualize_model_predictions, visualize_feature_maps, visualize_gradcam
from configs.config import load_config


def parse_args():
    """Parse command line arguments for evaluation"""
    parser = argparse.ArgumentParser(description='Breast Cancer Classification Evaluation')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the trained model checkpoint')
    parser.add_argument('--config-path', type=str, required=True,
                        help='Path to the configuration file')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Path to dataset (if different from config)')
    parser.add_argument('--results-dir', type=str, default=None,
                        help='Path to save evaluation results (if different from config)')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize model predictions and feature maps')
    parser.add_argument('--gradcam', action='store_true',
                        help='Generate Grad-CAM visualizations')
    parser.add_argument('--num-samples', type=int, default=6,
                        help='Number of samples to visualize')
    
    return parser.parse_args()


def main():
    """Main evaluation function"""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config_path)
    
    # Update configuration with command line arguments
    if args.data_dir:
        config['data']['data_dir'] = args.data_dir
    
    if args.results_dir:
        results_dir = args.results_dir
    else:
        results_dir = os.path.join(
            config['paths']['results_dir'],
            'evaluation'
        )
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    print("Creating data loaders...")
    _, _, test_loader, class_names = create_data_loaders(
        config['data']['data_dir'],
        img_size=config['data']['img_size'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        augment=False  # No need for augmentation during evaluation
    )
    print(f"Number of classes: {len(class_names)}")
    print(f"Class names: {class_names}")
    print(f"Testing samples: {len(test_loader.dataset)}")
    
    # Create model
    print(f"Creating {config['model']['model_name']} model...")
    model = get_model(
        config['model']['model_name'],
        num_classes=config['model']['num_classes'],
        pretrained=False,  # We'll load weights from checkpoint
        version=config['model']['version']
    )
    
    # Load model weights
    print(f"Loading model weights from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Test model
    print("Evaluating model on test set...")
    test_metrics, test_preds, test_labels, test_probs = test_model(
        model, test_loader, nn.CrossEntropyLoss(), device, class_names
    )
    
    # Print test metrics
    print("\nTest Metrics:")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1 Score: {test_metrics['f1']:.4f}")
    if test_metrics['auc'] > 0:
        print(f"AUC: {test_metrics['auc']:.4f}")
    
    # Print per-class metrics
    print("\nPer-Class Metrics:")
    for class_name, metrics in test_metrics['per_class'].items():
        print(f"{class_name}:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")
    
    # Plot confusion matrix
    plot_confusion_matrix(
        test_metrics['confusion_matrix'],
        class_names,
        os.path.join(results_dir, 'confusion_matrix.png')
    )
    
    # Plot ROC curve
    if config['model']['num_classes'] == 2:
        plot_roc_curve(
            test_labels,
            test_probs,
            class_names,
            os.path.join(results_dir, 'roc_curve.png')
        )
    
    # Save test metrics
    np.save(os.path.join(results_dir, 'test_metrics.npy'), test_metrics)
    
    # Visualizations
    if args.visualize:
        print("Generating visualizations...")
        
        # Visualize model predictions
        visualize_model_predictions(
            model,
            test_loader,
            class_names,
            device,
            num_images=args.num_samples,
            save_path=os.path.join(results_dir, 'model_predictions.png')
        )
        
        # Get a sample batch
        sample_images, sample_labels = next(iter(test_loader))
        
        # Visualize feature maps
        if 'densenet' in config['model']['model_name']:
            layer_name = 'features.denseblock1'
        elif 'resnet' in config['model']['model_name']:
            layer_name = 'base_model.4'  # ResNet's layer2
        elif 'efficientnet' in config['model']['model_name']:
            layer_name = 'features.2'
        else:
            layer_name = None
        
        if layer_name:
            visualize_feature_maps(
                model,
                sample_images[0].to(device),
                layer_name,
                device,
                save_path=os.path.join(results_dir, 'feature_maps.png')
            )
    
    # Grad-CAM visualizations
    if args.gradcam:
        print("Generating Grad-CAM visualizations...")
        
        # Get a sample batch
        sample_images, sample_labels = next(iter(test_loader))
        
        # Get target layer for Grad-CAM
        if 'densenet' in config['model']['model_name']:
            target_layer = model.features.denseblock4
        elif 'resnet' in config['model']['model_name']:
            target_layer = model.base_model[7]  # ResNet's last residual block
        elif 'efficientnet' in config['model']['model_name']:
            target_layer = model.features[-1]
        else:
            print("Grad-CAM not supported for this model architecture")
            return
        
        # Generate Grad-CAM for each class
        for class_idx in range(config['model']['num_classes']):
            visualize_gradcam(
                model,
                sample_images[0].to(device),
                target_layer,
                class_idx,
                device,
                save_path=os.path.join(results_dir, f'gradcam_class_{class_idx}.png')
            )
    
    print(f"Evaluation results saved to {results_dir}")


if __name__ == '__main__':
    main() 