import argparse
import os
import yaml
import torch

def get_default_config():
    """
    Get default configuration for the project
    
    Returns:
        config: Dictionary containing default configuration
    """
    config = {
        # Data configuration
        'data': {
            'data_dir': './data/breast_cancer_dataset',
            'img_size': 224,
            'batch_size': 32,
            'num_workers': 4,
            'augment': True
        },
        
        # Model configuration
        'model': {
            'model_name': 'densenet',  # 'densenet', 'resnet', 'efficientnet'
            'num_classes': 2,
            'pretrained': True,
            'version': None,  # '121', '50', 'b0'
        },
        
        # Training configuration
        'train': {
            'epochs': 30,
            'optimizer': 'adam',  # 'adam', 'sgd'
            'lr': 0.0001,
            'weight_decay': 1e-4,
            'patience': 10,
            'scheduler': 'plateau',  # 'plateau', 'step', None
            'seed': 42
        },
        
        # Paths
        'paths': {
            'model_dir': './models/checkpoints',
            'results_dir': './results',
            'log_dir': './logs'
        }
    }
    
    return config


def save_config(config, config_path):
    """
    Save configuration to a file
    
    Args:
        config: Configuration dictionary
        config_path: Path to save the configuration
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def load_config(config_path):
    """
    Load configuration from a file
    
    Args:
        config_path: Path to load the configuration from
        
    Returns:
        config: Loaded configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_train_args():
    """
    Parse command line arguments for training
    
    Returns:
        args: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Breast Cancer Classification Training')
    
    # Data args
    parser.add_argument('--data-dir', type=str, default='./data/breast_cancer_dataset',
                        help='Path to dataset')
    parser.add_argument('--img-size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training and validation')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--no-augment', action='store_false', dest='augment',
                        help='Disable data augmentation')
    
    # Model args
    parser.add_argument('--model-name', type=str, default='densenet',
                        choices=['densenet', 'resnet', 'efficientnet'],
                        help='Model architecture')
    parser.add_argument('--num-classes', type=int, default=2,
                        help='Number of classes')
    parser.add_argument('--no-pretrained', action='store_false', dest='pretrained',
                        help='Disable pretrained weights')
    parser.add_argument('--version', type=str, default=None,
                        help='Model version (e.g., 121 for DenseNet121)')
    
    # Training args
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs to train for')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'sgd'],
                        help='Optimizer for training')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--patience', type=int, default=10,
                        help='Patience for early stopping')
    parser.add_argument('--scheduler', type=str, default='plateau',
                        choices=['plateau', 'step', 'none'],
                        help='Learning rate scheduler')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Paths args
    parser.add_argument('--model-dir', type=str, default='./models/checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--results-dir', type=str, default='./results',
                        help='Directory to save results')
    parser.add_argument('--log-dir', type=str, default='./logs',
                        help='Directory for TensorBoard logs')
    
    # Config args
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration file')
    parser.add_argument('--save-config', type=str, default=None,
                        help='Path to save the configuration')
    
    args = parser.parse_args()
    
    # If config file is provided, load it and update args
    if args.config is not None:
        config = load_config(args.config)
        args = update_args_from_config(args, config)
    
    # Save configuration if requested
    if args.save_config is not None:
        config = args_to_config(args)
        save_config(config, args.save_config)
    
    return args


def update_args_from_config(args, config):
    """
    Update arguments from configuration dictionary
    
    Args:
        args: Parsed arguments
        config: Configuration dictionary
        
    Returns:
        args: Updated arguments
    """
    # Data args
    if 'data' in config:
        if 'data_dir' in config['data'] and args.data_dir == './data/breast_cancer_dataset':
            args.data_dir = config['data']['data_dir']
        if 'img_size' in config['data'] and args.img_size == 224:
            args.img_size = config['data']['img_size']
        if 'batch_size' in config['data'] and args.batch_size == 32:
            args.batch_size = config['data']['batch_size']
        if 'num_workers' in config['data'] and args.num_workers == 4:
            args.num_workers = config['data']['num_workers']
        if 'augment' in config['data']:
            args.augment = config['data']['augment']
    
    # Model args
    if 'model' in config:
        if 'model_name' in config['model'] and args.model_name == 'densenet':
            args.model_name = config['model']['model_name']
        if 'num_classes' in config['model'] and args.num_classes == 2:
            args.num_classes = config['model']['num_classes']
        if 'pretrained' in config['model']:
            args.pretrained = config['model']['pretrained']
        if 'version' in config['model'] and args.version is None:
            args.version = config['model']['version']
    
    # Training args
    if 'train' in config:
        if 'epochs' in config['train'] and args.epochs == 30:
            args.epochs = config['train']['epochs']
        if 'optimizer' in config['train'] and args.optimizer == 'adam':
            args.optimizer = config['train']['optimizer']
        if 'lr' in config['train'] and args.lr == 0.0001:
            args.lr = config['train']['lr']
        if 'weight_decay' in config['train'] and args.weight_decay == 1e-4:
            args.weight_decay = config['train']['weight_decay']
        if 'patience' in config['train'] and args.patience == 10:
            args.patience = config['train']['patience']
        if 'scheduler' in config['train'] and args.scheduler == 'plateau':
            args.scheduler = config['train']['scheduler']
        if 'seed' in config['train'] and args.seed == 42:
            args.seed = config['train']['seed']
    
    # Paths args
    if 'paths' in config:
        if 'model_dir' in config['paths'] and args.model_dir == './models/checkpoints':
            args.model_dir = config['paths']['model_dir']
        if 'results_dir' in config['paths'] and args.results_dir == './results':
            args.results_dir = config['paths']['results_dir']
        if 'log_dir' in config['paths'] and args.log_dir == './logs':
            args.log_dir = config['paths']['log_dir']
    
    return args


def args_to_config(args):
    """
    Convert arguments to configuration dictionary
    
    Args:
        args: Parsed arguments
        
    Returns:
        config: Configuration dictionary
    """
    config = {
        'data': {
            'data_dir': args.data_dir,
            'img_size': args.img_size,
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'augment': args.augment
        },
        'model': {
            'model_name': args.model_name,
            'num_classes': args.num_classes,
            'pretrained': args.pretrained,
            'version': args.version
        },
        'train': {
            'epochs': args.epochs,
            'optimizer': args.optimizer,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'patience': args.patience,
            'scheduler': args.scheduler,
            'seed': args.seed
        },
        'paths': {
            'model_dir': args.model_dir,
            'results_dir': args.results_dir,
            'log_dir': args.log_dir
        }
    }
    
    return config


def create_experiment_config(model_name, version=None, batch_size=32, lr=0.0001, epochs=30):
    """
    Create configuration for a specific experiment
    
    Args:
        model_name: Model name
        version: Model version
        batch_size: Batch size
        lr: Learning rate
        epochs: Number of epochs
        
    Returns:
        config: Configuration dictionary for the experiment
    """
    config = get_default_config()
    
    # Update config
    config['model']['model_name'] = model_name
    config['model']['version'] = version
    config['data']['batch_size'] = batch_size
    config['train']['lr'] = lr
    config['train']['epochs'] = epochs
    
    # Create experiment name
    experiment_name = f"{model_name}"
    if version:
        experiment_name += f"_{version}"
    experiment_name += f"_bs{batch_size}_lr{lr}_ep{epochs}"
    
    # Update paths
    config['paths']['model_dir'] = os.path.join(config['paths']['model_dir'], experiment_name)
    config['paths']['results_dir'] = os.path.join(config['paths']['results_dir'], experiment_name)
    config['paths']['log_dir'] = os.path.join(config['paths']['log_dir'], experiment_name)
    
    return config, experiment_name 