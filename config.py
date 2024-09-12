import os

class Config:
    # Data paths
    TRAIN_DIR = "./data/train"
    TEST_DIR = "./data/test"
    TRAIN_INFO_FILE = "./data/train.csv"
    TEST_INFO_FILE = "./data/test.csv"
    SAVE_RESULT_PATH = "./train_result"

    # Model settings
    MODEL_TYPE = 'timm'
    MODEL_NAME = 'resnet18'
    PRETRAINED = True

    # Training settings
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    EPOCHS = 5
    EPOCHS_PER_LR_DECAY = 2
    SCHEDULER_GAMMA = 0.1

    # Transform settings
    TRANSFORM_TYPE = "albumentations"

    # Ensure directories exist
    os.makedirs(SAVE_RESULT_PATH, exist_ok=True)