import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASET_NAME = "Nan-Do/code-search-net-python"
TRAIN_SIZE = 8000
VAL_SIZE = 1000
TEST_SIZE = 1000

MAX_SRC_LEN = 50
MAX_TGT_LEN = 80

EMBED_DIM = 256
HIDDEN_DIM = 256
NUM_LAYERS = 1
DROPOUT = 0.2

BATCH_SIZE = 32
EPOCHS = 10
LR = 0.001
TEACHER_FORCING_RATIO = 0.5

CHECKPOINT_DIR = "outputs/checkpoints"
PLOT_DIR = "outputs/plots"
LOG_DIR = "outputs/logs"
