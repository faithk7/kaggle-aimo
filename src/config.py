from pathlib import Path


class config:
    DATA_ROOT = Path("../data")
    MODEL_ROOT = Path("../models")
    TRAIN_PATH = DATA_ROOT / "train.csv"
    TEST_PATH = DATA_ROOT / "test.csv"

    CONFIG_DIR = Path("../config")

    DEBUG = True
    ENABLE_NEPTUNE = False
