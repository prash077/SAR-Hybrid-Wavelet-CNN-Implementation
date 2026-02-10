import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset")

TRAIN_DIR = os.path.join(DATA_DIR, "GroupB")
TEST_DIR = os.path.join(DATA_DIR, "GroupA")

MODEL_SAVE_PATH = os.path.join(BASE_DIR, "sar_model.pth")

BATCH_SIZE = 16
PATCH_SIZE = 64
EPOCHS = 3
LEARNING_RATE = 0.001
PATCHES_PER_FILE = 1000