import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize
import torch
from torchvision import transforms
import cv2
import os

def visualize_model_predictions(model, dataloader, class_names, device, num_images=6, save_path=None):
    """
    Visualize model predictions on a batch of images
    
    Args:
        model: PyTorch model
        dataloader: PyTorch DataLoader
        class_names: List of class names
        device: Device to run inference on
        num_images: Number of images to visualize
        save_path: Path to save the plot
    """
    model.eval()
    images, labels = next(iter(dataloader))
    
    # Get predictions
    with torch.no_grad():
        outputs = model(images.to(device))
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
    
    # Convert to numpy for visualization
    images = images.cpu().numpy()
    probs = probs.cpu().numpy()
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()
    
    # Plot images with predictions
    fig, axes = plt.subplots(2, 3, figsize=(15, 10)) if num_images >= 6 else plt.subplots(1, num_images, figsize=(15, 5))
    axes = axes.flatten() if num_images >= 6 else axes
    
    for i in range(min(num_images, len(images))):
        # Convert from CHW to HWC
        img = np.transpose(images[i], (1, 2, 0))
        
        # Denormalize
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        # Plot image
        axes[i].imshow(img)
        
        # Get prediction and confidence
        pred_class = class_names[preds[i]]
        pred_prob = probs[i][preds[i]]
        true_class = class_names[labels[i]]
        
        # Set title with prediction and confidence
        title = f"Pred: {pred_class} ({pred_prob:.2f})\nTrue: {true_class}"
        axes[i].set_title(title, color='green' if preds[i] == labels[i] else 'red')
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()


def visualize_feature_maps(model, image, layer_name, device, save_path=None):
    """
    Visualize feature maps of a specific layer for an input image
    
    Args:
        model: PyTorch model
        image: Input image tensor [1, C, H, W]
        layer_name: Name of the layer to visualize (e.g., 'features.denseblock1')
        device: Device to run inference on
        save_path: Path to save the plot
    """
    model.eval()
    image = image.unsqueeze(0).to(device) if image.dim() == 3 else image.to(device)
    
    # Register forward hook to get intermediate activations
    activations = {}
    
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach().cpu()
        return hook
    
    # Get the layer to visualize
    layer = None
    for name, module in model.named_modules():
        if name == layer_name:
            layer = module
            break
    
    if layer is None:
        print(f"Layer {layer_name} not found in model")
        return
    
    # Register hook
    handle = layer.register_forward_hook(get_activation(layer_name))
    
    # Forward pass
    with torch.no_grad():
        model(image)
    
    # Remove hook
    handle.remove()
    
    # Get activations
    feature_maps = activations[layer_name][0]
    
    # Visualize feature maps
    num_feature_maps = min(16, feature_maps.shape[0])
    fig, axes = plt.subplots(4, 4, figsize=(12, 12)) if num_feature_maps >= 16 else plt.subplots(2, num_feature_maps // 2, figsize=(12, 6))
    axes = axes.flatten()
    
    for i in range(num_feature_maps):
        feature_map = feature_maps[i].numpy()
        feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
        axes[i].imshow(feature_map, cmap='viridis')
        axes[i].set_title(f"Filter {i}")
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()


def visualize_gradcam(model, image, target_layer, class_idx, device, save_path=None):
    """
    Visualize Grad-CAM for a specific class
    
    Args:
        model: PyTorch model
        image: Input image tensor [C, H, W]
        target_layer: Target layer for Grad-CAM
        class_idx: Class index to generate Grad-CAM for
        device: Device to run inference on
        save_path: Path to save the plot
    """
    model.eval()
    image = image.unsqueeze(0).to(device) if image.dim() == 3 else image.to(device)
    image.requires_grad_()
    
    # Forward pass
    output = model(image)
    
    # Clear gradients
    model.zero_grad()
    
    # Backward pass with target class
    one_hot = torch.zeros_like(output)
    one_hot[0, class_idx] = 1
    output.backward(gradient=one_hot)
    
    # Get gradients and activations
    gradients = None
    activations = None
    
    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0].detach()
    
    def forward_hook(module, input, output):
        nonlocal activations
        activations = output.detach()
    
    # Register hooks
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)
    
    # Forward and backward pass
    output = model(image)
    one_hot = torch.zeros_like(output)
    one_hot[0, class_idx] = 1
    output.backward(gradient=one_hot)
    
    # Calculate Grad-CAM
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3]).cpu()
    activations = activations[0].cpu()
    
    for i in range(activations.shape[0]):
        activations[i] *= pooled_gradients[i]
    
    heatmap = torch.mean(activations, dim=0).numpy()
    heatmap = np.maximum(heatmap, 0)  # ReLU
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    # Convert image to numpy for visualization
    orig_img = image[0].detach().cpu().numpy().transpose(1, 2, 0)
    orig_img = orig_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    orig_img = np.clip(orig_img, 0, 1)
    
    # Resize heatmap to match image size
    heatmap = cv2.resize(heatmap, (orig_img.shape[1], orig_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    
    # Overlay heatmap on image
    superimposed_img = orig_img * 0.6 + heatmap * 0.4
    superimposed_img = np.clip(superimposed_img, 0, 1)
    
    # Plot original image and Grad-CAM
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(orig_img)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(superimposed_img)
    axes[1].set_title(f"Grad-CAM for Class {class_idx}")
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()


def visualize_comparison_all_metrics(models_metrics, model_names, save_dir=None):
    """
    Visualize comparison of all metrics between different models with multiple plots
    
    Args:
        models_metrics: List of metrics dictionaries for each model
        model_names: List of model names
        save_dir: Directory to save the plots
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    
    # Create save directory if it doesn't exist
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Bar plot for all metrics
    plt.figure(figsize=(14, 8))
    x = np.arange(len(model_names))
    width = 0.15
    
    for i, metric in enumerate(metrics):
        values = [metrics_dict[metric] for metrics_dict in models_metrics]
        plt.bar(x + i*width - 0.3, values, width, label=metric.capitalize())
    
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Comparison of Model Performance')
    plt.xticks(x, model_names)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'all_metrics_comparison.png'))
    
    plt.show()
    
    # Individual bar plots for each metric
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        values = [metrics_dict[metric] for metrics_dict in models_metrics]
        
        bars = plt.bar(model_names, values, color='skyblue')
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
        
        plt.xlabel('Models')
        plt.ylabel(f'{metric.capitalize()} Score')
        plt.title(f'Comparison of {metric.capitalize()} Score')
        plt.ylim(0, 1.1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, f'{metric}_comparison.png'))
        
        plt.show()


def visualize_model_architecture(model, save_path=None):
    """
    Visualize model architecture as a table
    
    Args:
        model: PyTorch model
        save_path: Path to save the plot
    """
    from textwrap import wrap
    
    # Get model layers and parameters
    layers = []
    total_params = 0
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Only leaf modules
            # Count parameters
            param_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
            total_params += param_count
            
            # Get layer type and shape
            layer_type = module.__class__.__name__
            shape = 'N/A'
            
            if hasattr(module, 'weight') and hasattr(module.weight, 'shape'):
                shape = str(tuple(module.weight.shape))
            
            layers.append({
                'name': name,
                'type': layer_type,
                'parameters': param_count,
                'shape': shape
            })
    
    # Create a table
    fig, ax = plt.subplots(figsize=(12, len(layers) * 0.3 + 1))
    ax.axis('tight')
    ax.axis('off')
    
    # Wrap long names
    wrapped_names = ['\n'.join(wrap(name, 40)) for name in [layer['name'] for layer in layers]]
    
    table_data = [
        [layer['type'], wrapped_name, f"{layer['parameters']:,}", layer['shape']]
        for layer, wrapped_name in zip(layers, wrapped_names)
    ]
    
    table = ax.table(
        cellText=table_data,
        colLabels=['Layer Type', 'Name', 'Parameters', 'Shape'],
        loc='center',
        cellLoc='center'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    plt.title(f"Model Architecture - Total Parameters: {total_params:,}", pad=20)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    plt.show()
    
    return total_params 