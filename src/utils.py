import argparse
import time

import yaml

from config import config


def time_it(func):
    """A decorator that prints the execution time of the function it decorates."""

    def wrapper(*args, **kwargs):
        start_time = time.time()  # Capture the start time
        result = func(*args, **kwargs)  # Execute the function
        end_time = time.time()  # Capture the end time
        print(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result

    return wrapper


def load_config(config_filename: str) -> None:
    config_path = config.CONFIG_DIR / config_filename

    with open(config_path, "r") as file:
        yaml_config = yaml.safe_load(file)

    for key, value in yaml_config.items():
        setattr(config, key, value)


def parse_args():
    parser = argparse.ArgumentParser(description="Update config values at runtime.")
    parser.add_argument(
        "--config_path", "-c", type=str, help="Relative path to YAML config file"
    )

    return parser.parse_args()
