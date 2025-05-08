import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from data.dataset import create_data_loaders, visualize_batch
from models.models import get_model
from utils.train_utils import train_model, test_model, plot_training_history, plot_confusion_matrix, plot_roc_curve
from utils.visualization import visualize_model_predictions, visualize_model_architecture
from configs.config import get_train_args, save_config, args_to_config


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    # Parse arguments
    args = get_train_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create directories
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Save configuration
    config = args_to_config(args)
    save_config(config, os.path.join(args.model_dir, 'config.yaml'))
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader, class_names = create_data_loaders(
        args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=args.augment
    )
    print(f"Number of classes: {len(class_names)}")
    print(f"Class names: {class_names}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Testing samples: {len(test_loader.dataset)}")
    
    # Visualize a batch of images
    visualize_batch(train_loader, class_names)
    
    # Create model
    print(f"Creating {args.model_name} model...")
    model = get_model(
        args.model_name,
        num_classes=args.num_classes,
        pretrained=args.pretrained,
        version=args.version
    )
    model = model.to(device)
    
    # Visualize model architecture
    visualize_model_architecture(model, os.path.join(args.results_dir, 'model_architecture.png'))
    
    # Create TensorBoard writer
    writer = SummaryWriter(args.log_dir)
    
    # Train model
    print("Training model...")
    model, history = train_model(model, train_loader, val_loader, args, device)
    
    # Plot training history
    plot_training_history(history, os.path.join(args.results_dir, 'training_history.png'))
    
    # Test model
    print("Testing model...")
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
        os.path.join(args.results_dir, 'confusion_matrix.png')
    )
    
    # Plot ROC curve
    if args.num_classes == 2:
        plot_roc_curve(
            test_labels,
            test_probs,
            class_names,
            os.path.join(args.results_dir, 'roc_curve.png')
        )
    
    # Visualize model predictions
    visualize_model_predictions(
        model,
        test_loader,
        class_names,
        device,
        num_images=6,
        save_path=os.path.join(args.results_dir, 'model_predictions.png')
    )
    
    # Save test metrics
    np.save(os.path.join(args.results_dir, 'test_metrics.npy'), test_metrics)
    
    # Close TensorBoard writer
    writer.close()
    
    print(f"Results saved to {args.results_dir}")


def train_multiple_models():
    """Train multiple models for comparison"""
    # Model configurations
    model_configs = [
        {'model_name': 'densenet', 'version': '121'},
        {'model_name': 'resnet', 'version': '50'},
        {'model_name': 'efficientnet', 'version': 'b0'}
    ]
    
    # Parse base arguments
    base_args = get_train_args()
    
    # Set random seed
    set_seed(base_args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders (once)
    print("Creating data loaders...")
    train_loader, val_loader, test_loader, class_names = create_data_loaders(
        base_args.data_dir,
        img_size=base_args.img_size,
        batch_size=base_args.batch_size,
        num_workers=base_args.num_workers,
        augment=base_args.augment
    )
    
    # Initialize results
    all_metrics = []
    model_names = []
    
    # Train each model
    for config in model_configs:
        # Update args for this model
        args = base_args
        args.model_name = config['model_name']
        args.version = config['version']
        
        # Create model-specific directories
        model_subdir = f"{args.model_name}_{args.version}"
        args.model_dir = os.path.join(base_args.model_dir, model_subdir)
        args.results_dir = os.path.join(base_args.results_dir, model_subdir)
        args.log_dir = os.path.join(base_args.log_dir, model_subdir)
        
        os.makedirs(args.model_dir, exist_ok=True)
        os.makedirs(args.results_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)
        
        # Save configuration
        model_config = args_to_config(args)
        save_config(model_config, os.path.join(args.model_dir, 'config.yaml'))
        
        # Create model
        print(f"\nCreating {args.model_name}_{args.version} model...")
        model = get_model(
            args.model_name,
            num_classes=args.num_classes,
            pretrained=args.pretrained,
            version=args.version
        )
        model = model.to(device)
        
        # Create TensorBoard writer
        writer = SummaryWriter(args.log_dir)
        
        # Train model
        print(f"Training {args.model_name}_{args.version} model...")
        model, history = train_model(model, train_loader, val_loader, args, device)
        
        # Plot training history
        plot_training_history(history, os.path.join(args.results_dir, 'training_history.png'))
        
        # Test model
        print(f"Testing {args.model_name}_{args.version} model...")
        test_metrics, test_preds, test_labels, test_probs = test_model(
            model, test_loader, nn.CrossEntropyLoss(), device, class_names
        )
        
        # Print test metrics
        print(f"\n{args.model_name}_{args.version} Test Metrics:")
        print(f"Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Precision: {test_metrics['precision']:.4f}")
        print(f"Recall: {test_metrics['recall']:.4f}")
        print(f"F1 Score: {test_metrics['f1']:.4f}")
        if test_metrics['auc'] > 0:
            print(f"AUC: {test_metrics['auc']:.4f}")
        
        # Plot confusion matrix
        plot_confusion_matrix(
            test_metrics['confusion_matrix'],
            class_names,
            os.path.join(args.results_dir, 'confusion_matrix.png')
        )
        
        # Save test metrics
        np.save(os.path.join(args.results_dir, 'test_metrics.npy'), test_metrics)
        
        # Store metrics for comparison
        all_metrics.append({
            'accuracy': test_metrics['accuracy'],
            'precision': test_metrics['precision'],
            'recall': test_metrics['recall'],
            'f1': test_metrics['f1'],
            'auc': test_metrics['auc']
        })
        model_names.append(f"{args.model_name}_{args.version}")
        
        # Close TensorBoard writer
        writer.close()
    
    # Compare models
    from utils.visualization import visualize_comparison_all_metrics
    
    # Create comparison directory
    comparison_dir = os.path.join(base_args.results_dir, 'comparison')
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Visualize comparison
    visualize_comparison_all_metrics(
        all_metrics,
        model_names,
        save_dir=comparison_dir
    )
    
    print(f"Comparison results saved to {comparison_dir}")


if __name__ == '__main__':
    main()
    # Uncomment the following line to train and compare multiple models
    # train_multiple_models() 