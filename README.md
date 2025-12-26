# Malware Detection Using DCT-Based Image Visualization

A deep learning approach for malware detection using frequency domain image visualization.

## Pipelines

| Pipeline | Method                | Input                      | Accuracy |
| -------- | --------------------- | -------------------------- | -------- |
| 1        | Bigram-DCT            | Single-channel (256×256×1) | ~94-95%  |
| 2        | Byteplot + Bigram-DCT | Two-channel (256×256×2)    | ~96%     |

## Setup

```bash
pip install -r requirements.txt
python test_suite.py  # Verify installation
```

## Data Structure

```
data/
├── malware/    # Malicious executables
└── benign/     # Clean executables
```

## Usage

1. Edit configuration in `main.py`:

```python
PIPELINE = 1              # 1 or 2
DATA_DIR = "./data"
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DEVICE = "auto"           # auto, cpu, cuda
```

2. Run training:

```bash
python main.py
```

3. Test a saved model:

```python
MODEL_PATH = "./checkpoints/pipeline1_best.pth"
TEST_ONLY = True
```

## Configuration

| Parameter        | Default         | Description               |
| ---------------- | --------------- | ------------------------- |
| `PIPELINE`       | `1`             | Pipeline to use (1 or 2)  |
| `DATA_DIR`       | `./data`        | Dataset directory         |
| `MAX_SAMPLES`    | `None`          | Limit samples for testing |
| `EPOCHS`         | `50`            | Training epochs           |
| `BATCH_SIZE`     | `32`            | Batch size                |
| `LEARNING_RATE`  | `0.001`         | Learning rate             |
| `PATIENCE`       | `10`            | Early stopping patience   |
| `TRAIN_SPLIT`    | `0.7`           | Training split            |
| `VAL_SPLIT`      | `0.2`           | Validation split          |
| `TEST_SPLIT`     | `0.1`           | Test split                |
| `CHECKPOINT_DIR` | `./checkpoints` | Model save path           |
| `MODEL_PATH`     | `None`          | Pre-trained model path    |
| `TEST_ONLY`      | `False`         | Skip training             |
| `OUTPUT_DIR`     | `./results`     | Results directory         |
| `NO_PLOT`        | `False`         | Disable plots             |
| `DEVICE`         | `auto`          | Device selection          |

## Output

- `checkpoints/pipeline{1,2}_best.pth` - Trained model
- `results/pipeline{1,2}_training_history.png` - Training curves
- `results/pipeline{1,2}_roc_curve.png` - ROC curve
- `results/pipeline{1,2}_confusion_matrix.png` - Confusion matrix

## Requirements

- Python 3.7+
- PyTorch >= 1.12.0
- NumPy, SciPy, scikit-learn, matplotlib
