from pathlib import Path


class CONFIG:
    # == general ==
    DEBUG = True
    ENABLE_NEPTUNE = False

    # == path ==
    DATA_ROOT = Path("../data")
    MODEL_ROOT = Path("../models")
    TRAIN_PATH = DATA_ROOT / "train.csv"
    TEST_PATH = DATA_ROOT / "test.csv"

    CONFIG_DIR = Path("../config")
    CACHE_DIR = Path("../tmp")

    # == training  ==
    MAX_LENGTH = 512
    NUM_TRAIN_EPOCHS = 3
    PER_DEVICE_TRAIN_BATCH_SIZE = 4
    PER_DEVICE_EVAL_BATCH_SIZE = 4
    WARMUP_STEPS = 500
    WEIGHT_DECAY = 0.01
    LOGGING_DIR = Path("./logs")
    LOGGING_STEPS = 10
    MODEL = "deepseek-ai/deepseek-math-7b-base"
