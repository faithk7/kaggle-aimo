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


# TODO: combine the below two functions into one
def load_config_from_yaml(yaml_relative_path: str) -> None:
    yaml_path = config.CONFIG_DIR / yaml_relative_path

    with open(yaml_path, "r") as file:
        yaml_config = yaml.safe_load(file)

    for key, value in yaml_config.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Invalid config key: {key} when loading from yaml")


def load_config_from_args(args: argparse.Namespace) -> None:
    for arg in vars(args):
        value = getattr(args, arg)
        if value is not None:
            setattr(config, arg, value)
        else:
            raise ValueError(f"Invalid config key: {arg} when loading from args")


def parse_args():
    parser = argparse.ArgumentParser(description="Update config values at runtime.")
    parser.add_argument(
        "--config_path", "-c", type=str, help="Relative path to YAML config file"
    )

    return parser.parse_args()
