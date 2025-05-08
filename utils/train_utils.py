import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train model for one epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    # Use tqdm for progress bar
    loop = tqdm(dataloader, desc="Training")
    
    for inputs, labels in loop:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Update running loss
        running_loss += loss.item() * inputs.size(0)
        
        # Save predictions and labels
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        loop.set_postfix(loss=loss.item())
    
    # Calculate metrics
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return epoch_loss, epoch_acc, epoch_f1, all_preds, all_labels


def validate(model, dataloader, criterion, device):
    """Validate model on validation set"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []  # For ROC-AUC
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Update running loss
            running_loss += loss.item() * inputs.size(0)
            
            # Get predictions
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            # Save predictions, probabilities, and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    val_loss = running_loss / len(dataloader.dataset)
    val_acc = accuracy_score(all_labels, all_preds)
    val_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    val_recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    val_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    # For binary classification, calculate ROC-AUC
    val_auc = 0
    if len(np.unique(all_labels)) == 2:
        val_auc = roc_auc_score(all_labels, [prob[1] for prob in all_probs])
    
    metrics = {
        'loss': val_loss,
        'accuracy': val_acc,
        'precision': val_precision,
        'recall': val_recall,
        'f1': val_f1,
        'auc': val_auc
    }
    
    return metrics, all_preds, all_labels


def test_model(model, dataloader, criterion, device, class_names):
    """Test model on test set and return detailed metrics"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Get predictions
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            # Save predictions, probabilities, and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    test_acc = accuracy_score(all_labels, all_preds)
    test_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    test_recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    test_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    # For binary classification, calculate ROC-AUC
    test_auc = 0
    if len(np.unique(all_labels)) == 2:
        test_auc = roc_auc_score(all_labels, [prob[1] for prob in all_probs])
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Calculate per-class precision, recall, and F1
    class_metrics = {}
    for i, class_name in enumerate(class_names):
        if i in np.unique(all_labels):  # Only calculate for classes that are present
            class_precision = precision_score(all_labels, all_preds, labels=[i], average=None, zero_division=0)[0]
            class_recall = recall_score(all_labels, all_preds, labels=[i], average=None, zero_division=0)[0]
            class_f1 = f1_score(all_labels, all_preds, labels=[i], average=None, zero_division=0)[0]
            
            class_metrics[class_name] = {
                'precision': class_precision,
                'recall': class_recall,
                'f1': class_f1
            }
    
    metrics = {
        'accuracy': test_acc,
        'precision': test_precision,
        'recall': test_recall,
        'f1': test_f1,
        'auc': test_auc,
        'confusion_matrix': cm,
        'per_class': class_metrics
    }
    
    return metrics, all_preds, all_labels, all_probs


def train_model(model, train_loader, val_loader, args, device):
    """
    Train a model with given parameters
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        args: Training arguments
        device: Device to train on
        
    Returns:
        model: Trained model
        history: Training history
    """
    # Set up criterion, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Initialize variables for early stopping
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    # Initialize history dictionary
    history = {
        'train_loss': [],
        'train_acc': [],
        'train_f1': [],
        'val_loss': [],
        'val_acc': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'val_auc': [],
        'lr': []
    }
    
    # Train for specified number of epochs
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        # Train one epoch
        train_loss, train_acc, train_f1, _, _ = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_metrics, _, _ = validate(model, val_loader, criterion, device)
        
        # Update learning rate if needed
        scheduler.step(val_metrics['loss'])
        
        # Save metrics to history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_auc'].append(val_metrics['auc'])
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['f1']:.4f}")
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_epoch = epoch
            patience_counter = 0
            
            # Save model
            os.makedirs(args.model_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_acc': val_metrics['accuracy'],
                'val_f1': val_metrics['f1']
            }, os.path.join(args.model_dir, f"{args.model_name}_best.pth"))
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"Early stopping triggered after {epoch+1} epochs. Best epoch: {best_epoch+1}")
            break
    
    # Load best model
    checkpoint = torch.load(os.path.join(args.model_dir, f"{args.model_name}_best.pth"))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, history


def plot_training_history(history, save_path=None):
    """
    Plot training history
    
    Args:
        history: Training history dictionary
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss plot
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    
    # Accuracy plot
    axes[0, 1].plot(history['train_acc'], label='Train Accuracy')
    axes[0, 1].plot(history['val_acc'], label='Validation Accuracy')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    
    # F1 Score plot
    axes[1, 0].plot(history['train_f1'], label='Train F1')
    axes[1, 0].plot(history['val_f1'], label='Validation F1')
    axes[1, 0].set_title('F1 Score')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].legend()
    
    # Learning Rate plot
    axes[1, 1].plot(history['lr'])
    axes[1, 1].set_title('Learning Rate')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()


def plot_confusion_matrix(cm, class_names, save_path=None):
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()


def plot_roc_curve(labels, probs, class_names, save_path=None):
    """
    Plot ROC curve
    
    Args:
        labels: True labels
        probs: Predicted probabilities
        class_names: List of class names
        save_path: Path to save the plot
    """
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    import numpy as np
    
    # Binarize labels for one-vs-rest ROC
    n_classes = len(class_names)
    
    if n_classes == 2:
        # Binary classification
        fpr, tpr, _ = roc_curve(labels, [p[1] for p in probs])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc='lower right')
    else:
        # Multi-class classification
        y_bin = label_binarize(labels, classes=range(n_classes))
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], [p[i] for p in probs])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Plot ROC curves
        plt.figure(figsize=(10, 8))
        
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], lw=2, 
                     label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) - One vs Rest')
        plt.legend(loc='lower right')
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()


def plot_comparison_metrics(models_metrics, model_names, save_path=None):
    """
    Plot comparison of metrics between different models
    
    Args:
        models_metrics: List of metrics dictionaries for each model
        model_names: List of model names
        save_path: Path to save the plot
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    metrics_values = {metric: [] for metric in metrics}
    
    # Extract metrics for each model
    for metrics_dict in models_metrics:
        for metric in metrics:
            metrics_values[metric].append(metrics_dict[metric])
    
    # Plot
    plt.figure(figsize=(12, 8))
    x = np.arange(len(model_names))
    width = 0.15
    
    for i, metric in enumerate(metrics):
        plt.bar(x + i*width - 0.3, metrics_values[metric], width, label=metric.capitalize())
    
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Comparison of Model Performance')
    plt.xticks(x, model_names)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show() 