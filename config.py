DATA_CONFIG = {
    'data_dir': './data',
    'train_split': 0.7,
    'val_split': 0.2,
    'test_split': 0.1,
    'max_samples': None,  # None for all samples
    'num_workers': 2,     # Number of data loading workers
}

MODEL_CONFIG = {
    'image_size': 256,
    
    'conv_channels': [32, 64, 128],
    'kernel_size': 3,
    'pool_size': 2,
    'fc_layers': [512, 256],
    'dropout_rate': 0.5,
    
    'input_channels': 1,  # XOR Model (Single Channel)
}

TRAINING_CONFIG = {
    'batch_size': 32,
    'num_epochs': 50,
    'learning_rate': 0.001,
    'patience': 10,  # Early stopping patience
    'device': 'auto',  # 'auto', 'cpu', or 'cuda'
}

IMAGE_CONFIG = {
    'zero_out_bigram_0000': True,  # As per paper
    'bigram_normalization': True,
    
    'dct_normalization': 'ortho',
    
    'byteplot_resize_method': 'bilinear',
}

PATH_CONFIG = {
    'checkpoint_dir': './checkpoints',
    'output_dir': './results',
    'log_dir': './logs',
}

EVAL_CONFIG = {
    'plot_training_history': True,
    'plot_roc_curve': True,
    'plot_confusion_matrix': True,
    'save_predictions': False,
    'prediction_threshold': 0.5,
}

LOG_CONFIG = {
    'log_level': 'INFO',
    'log_to_file': False,
    'log_file': './logs/training.log',
}


def get_config(pipeline: int = 2):
    """
    Get complete configuration.
    
    Args:
        pipeline: Kept for compatibility, always uses XOR model
        
    Returns:
        Dictionary with all configuration parameters
    """
    config = {
        'pipeline': 2, # Always pipeline 2 (XOR)
        'data': DATA_CONFIG.copy(),
        'model': MODEL_CONFIG.copy(),
        'training': TRAINING_CONFIG.copy(),
        'image': IMAGE_CONFIG.copy(),
        'paths': PATH_CONFIG.copy(),
        'eval': EVAL_CONFIG.copy(),
        'log': LOG_CONFIG.copy(),
    }
    
    config['model']['input_channels'] = MODEL_CONFIG['input_channels']
    
    return config


def print_config(config):
    for section, params in config.items():
        if isinstance(params, dict):
            print(f"\n{section.upper()}:")
            for key, value in params.items():
                print(f"  {key}: {value}")
        else:
            print(f"{section}: {params}")
    

# Example usage
if __name__ == "__main__":
    print("XOR Model Configuration:")
    config = get_config(pipeline=2)
    print_config(config)
