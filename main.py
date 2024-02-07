import argparse

from mains import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, help="The task to perform.")
    parser.add_argument("--config_path", type=str, help="Path to the appropriate .yaml config file.")
    args = parser.parse_args()

    task = args.task
    config_path = args.config_path

    if task == "preprocess":
        preprocessor_main(config_path)
    elif task == "train_detector":
        detector_training_main(config_path)
