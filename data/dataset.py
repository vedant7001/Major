import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class BreastCancerDataset(Dataset):
    def __init__(self, image_paths, labels=None, transform=None):
        """
        Breast Cancer Classification Dataset
        Args:
            image_paths (list): List of paths to images
            labels (list): List of labels (0: normal, 1: benign, 2: malignant)
            transform (callable, optional): Transform to be applied on images
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        if self.labels is not None:
            label = self.labels[idx]
            return image, label
        else:
            return image


def create_data_loaders(data_dir, img_size=224, batch_size=32, num_workers=4, augment=True):
    """
    Create train, validation, and test data loaders
    
    Args:
        data_dir (str): Directory containing the dataset
        img_size (int): Size to resize images to
        batch_size (int): Batch size for DataLoader
        num_workers (int): Number of workers for DataLoader
        augment (bool): Whether to apply data augmentation on training set
        
    Returns:
        train_loader, val_loader, test_loader: PyTorch DataLoaders
        class_names (list): List of class names
    """
    # Define transformations
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]) if augment else transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Get all image paths and labels
    image_paths = []
    labels = []
    class_names = []
    
    # For standard dataset format where each class has its own directory
    for class_idx, class_name in enumerate(sorted(os.listdir(data_dir))):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        class_names.append(class_name)
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(class_dir, img_name))
                labels.append(class_idx)
    
    # Split into train, validation, and test sets (70/15/15 split)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    
    # Create datasets
    train_dataset = BreastCancerDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = BreastCancerDataset(val_paths, val_labels, transform=val_transform)
    test_dataset = BreastCancerDataset(test_paths, test_labels, transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, class_names


def visualize_batch(dataloader, class_names, num_images=5):
    """
    Visualize a batch of images from the dataset
    
    Args:
        dataloader: PyTorch DataLoader
        class_names (list): List of class names
        num_images (int): Number of images to visualize
    """
    images, labels = next(iter(dataloader))
    
    # Convert images from tensor to numpy for visualization
    images = images.numpy()
    
    # Plot images
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    for i in range(num_images):
        # Convert from CHW to HWC
        img = np.transpose(images[i], (1, 2, 0))
        # Denormalize
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        axes[i].set_title(class_names[labels[i]])
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show() 