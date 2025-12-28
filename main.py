import os
import torch
from pathlib import Path

from config import get_config
from utils.data_loader import create_data_loaders
from models.cnn_models import C3C2D_TwoChannel, count_parameters
from utils.training import (
    train_model, 
    test_model, 
    plot_training_history,
    plot_roc_curve,
    plot_confusion_matrix
)

def setup_directories(config):
    Path(config['paths']['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['paths']['output_dir']).mkdir(parents=True, exist_ok=True)


def get_device(device_arg: str) -> torch.device:
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    return device


def run_pipeline(config):
    pipeline_id = config['pipeline']
    print(f"Pipeline {pipeline_id}: XOR Model (Byteplot + Bigram-DCT)")
    print()
    
    device = get_device(config['training']['device'])
    setup_directories(config)
    
    print("Loading data...")
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=config['data']['data_dir'],
        mode='two_channel',
        batch_size=config['training']['batch_size'],
        train_split=config['data']['train_split'],
        val_split=config['data']['val_split'],
        test_split=config['data']['test_split'],
        max_samples=config['data']['max_samples'],
        num_workers=config['data']['num_workers']
    )
    
    print("\nInitializing model...")
    model = C3C2D_TwoChannel()
    print(f"Model: 3C2D CNN (XOR Model)")

    print(f"Total parameters: {count_parameters(model):,}")
    
    model_save_path = os.path.join(config['paths']['checkpoint_dir'], f'pipeline{pipeline_id}_best.pth')
    
    # Training
    if not config.get('test_only', False):
        print("Training...")        
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config['training']['num_epochs'],
            learning_rate=config['training']['learning_rate'],
            device=device,
            save_path=model_save_path,
            patience=config['training']['patience']
        )
        
        if config['eval']['plot_training_history']:
            plot_path = os.path.join(config['paths']['output_dir'], f'pipeline{pipeline_id}_training_history.png')
            plot_training_history(history, save_path=plot_path)
    
    print("Testing...")    
    metrics = test_model(model, test_loader, device)
    
    if config['eval']['plot_roc_curve']:
        roc_path = os.path.join(config['paths']['output_dir'], f'pipeline{pipeline_id}_roc_curve.png')
        plot_roc_curve(metrics, save_path=roc_path)
        
    if config['eval']['plot_confusion_matrix']:
        cm_path = os.path.join(config['paths']['output_dir'], f'pipeline{pipeline_id}_confusion_matrix.png')
        plot_confusion_matrix(metrics['confusion_matrix'], save_path=cm_path)
    
    return metrics


def main():
    SELECTED_PIPELINE = 2
    
    config = get_config(pipeline=SELECTED_PIPELINE)
    
    print("\nConfiguration:")
    print(f"\tPipeline:       {config['pipeline']}")
    print(f"\tData directory: {config['data']['data_dir']}")
    print(f"\tBatch size:     {config['training']['batch_size']}")
    print(f"\tEpochs:         {config['training']['num_epochs']}")
    print(f"\tLearning rate:  {config['training']['learning_rate']}")
    print(f"\tDevice:         {config['training']['device']}")
    print()
    
    metrics = run_pipeline(config)
    
    print(f"\nPipeline {config['pipeline']} Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"Pipeline {config['pipeline']} AUC:      {metrics.get('auc', 0):.4f}")
    print(f"Results saved to: {config['paths']['output_dir']}")
    print(f"Model saved to:   {config['paths']['checkpoint_dir']}")


if __name__ == "__main__":
    main()
