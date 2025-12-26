import os
import torch
from pathlib import Path

from utils.data_loader import create_data_loaders
from models.cnn_models import C3C2D_SingleChannel, C3C2D_TwoChannel, count_parameters
from utils.training import (
    train_model, 
    test_model, 
    plot_training_history,
    plot_roc_curve,
    plot_confusion_matrix
)



# Pipeline selection: 1 (Bigram-DCT single-channel) or 2 (Two-channel ensemble)
PIPELINE = 1

# Data settings
DATA_DIR = "./data"
MAX_SAMPLES = None

EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
PATIENCE = 10

TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.2
TEST_SPLIT = 0.1

# Model saving/loading
CHECKPOINT_DIR = "./checkpoints"
MODEL_PATH = None
TEST_ONLY = False  # Set to True to only run testing (requires MODEL_PATH)

OUTPUT_DIR = "./results"
NO_PLOT = False  # True to skip plotting

# Device: 'cpu', or 'cuda'
DEVICE = "auto"


def setup_directories(checkpoint_dir: str, output_dir: str):
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)


def get_device(device_arg: str) -> torch.device:
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    return device


def run_pipeline_1():
    print("Pipeline 1: BIGRAM-DCT Frequency Image (Single-Channel)")
    print()
    
    device = get_device(DEVICE)
    setup_directories(CHECKPOINT_DIR, OUTPUT_DIR)
    
    print("Loading data...")
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=DATA_DIR,
        mode='bigram_dct',
        batch_size=BATCH_SIZE,
        train_split=TRAIN_SPLIT,
        val_split=VAL_SPLIT,
        test_split=TEST_SPLIT,
        max_samples=MAX_SAMPLES,
        num_workers=0
    )
    
    print("\nmodel...")
    model = C3C2D_SingleChannel()
    print(f"Model: 3C2D CNN (Single-Channel)")
    print(f"Total parameters: {count_parameters(model):,}")
    
    model_save_path = os.path.join(CHECKPOINT_DIR, 'pipeline1_best.pth')
    
    # Load model if specified
    if MODEL_PATH:
        print(f"\nLoading model from {MODEL_PATH}")
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Training
    if not TEST_ONLY:
        print("Training...")
        
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            device=device,
            save_path=model_save_path,
            patience=PATIENCE
        )
        
        if not NO_PLOT:
            plot_path = os.path.join(OUTPUT_DIR, 'pipeline1_training_history.png')
            plot_training_history(history, save_path=plot_path)
    
    print("Testing...")
    
    metrics = test_model(model, test_loader, device)
    
    # Plot results
    if not NO_PLOT:
        # ROC
        roc_path = os.path.join(OUTPUT_DIR, 'pipeline1_roc_curve.png')
        plot_roc_curve(metrics, save_path=roc_path)
        
        # Confusion matrix
        cm_path = os.path.join(OUTPUT_DIR, 'pipeline1_confusion_matrix.png')
        plot_confusion_matrix(metrics['confusion_matrix'], save_path=cm_path)
    
    return metrics


def run_pipeline_2():
    print("Pipeline 2: Ensemble Model (Byteplot + Bigram-DCT)")
    print()
    
    device = get_device(DEVICE)
    setup_directories(CHECKPOINT_DIR, OUTPUT_DIR)
    
    print("Loading data...")
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=DATA_DIR,
        mode='two_channel',
        batch_size=BATCH_SIZE,
        train_split=TRAIN_SPLIT,
        val_split=VAL_SPLIT,
        test_split=TEST_SPLIT,
        max_samples=MAX_SAMPLES,
        num_workers=0
    )
    
    print("\nInitializing model...")
    model = C3C2D_TwoChannel()
    print(f"Model: 3C2D CNN (Two-Channel)")
    print(f"Total parameters: {count_parameters(model):,}")
    
    model_save_path = os.path.join(CHECKPOINT_DIR, 'pipeline2_best.pth')
    
    if MODEL_PATH:
        print(f"\nLoading model from {MODEL_PATH}")
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Training
    if not TEST_ONLY:
        print("Training...")
        
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            device=device,
            save_path=model_save_path,
            patience=PATIENCE
        )
        
        if not NO_PLOT:
            plot_path = os.path.join(OUTPUT_DIR, 'pipeline2_training_history.png')
            plot_training_history(history, save_path=plot_path)
    
    print("Testing...")
    
    metrics = test_model(model, test_loader, device)
    
    if not NO_PLOT:
        # ROC
        roc_path = os.path.join(OUTPUT_DIR, 'pipeline2_roc_curve.png')
        plot_roc_curve(metrics, save_path=roc_path)
        
        # Confusion matrix
        cm_path = os.path.join(OUTPUT_DIR, 'pipeline2_confusion_matrix.png')
        plot_confusion_matrix(metrics['confusion_matrix'], save_path=cm_path)
    
    return metrics


def main():
    if TEST_ONLY and MODEL_PATH is None:
        raise ValueError("TEST_ONLY requires MODEL_PATH to be set")

    print("\nConfiguration:")
    print(f"\tPipeline:       {PIPELINE}")
    print(f"\tData directory: {DATA_DIR}")
    print(f"\tBatch size:     {BATCH_SIZE}")
    print(f"\tEpochs:         {EPOCHS}")
    print(f"\tLearning rate:  {LEARNING_RATE}")
    print(f"\tDevice:         {DEVICE}")
    print(f"\tTest only:      {TEST_ONLY}")
    print()
    
    if PIPELINE == 1:
        metrics = run_pipeline_1()
    elif PIPELINE == 2:
        metrics = run_pipeline_2()
    else:
        raise ValueError(f"Invalid PIPELINE value: {PIPELINE}. Must be 1 or 2.")
    
    print(f"\nPipeline {PIPELINE} Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"Pipeline {PIPELINE} AUC:      {metrics.get('auc', 0):.4f}")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"Model saved to:   {CHECKPOINT_DIR}")


if __name__ == "__main__":
    main()
