#!/usr/bin/python3
'''Contains utility code such as loggers, file opening operations, etc.
'''
import os
import logging
import numpy as np
import matplotlib.pyplot as plt

import yaml
import csv


def setup_log(log_path, log_name, level):
    '''Configures logging.

    Args:
        log_path (str): Location of log file.
        log_name (str): Log file prefix (will be followed by timestamp and PID).
        level (str): noset|debug|info|warning|error|critical
    '''
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    filename = os.path.join(f"{log_path}", (f'{log_name}.log'))

    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=getattr(logging, level.upper()),
        handlers=[
            logging.FileHandler(filename, mode="w"),
            logging.StreamHandler()
        ]
    )


def load_yaml(yaml_path):
    '''Import YAML file.

    Args:
        yaml_path (str): Path to YAML file.

    Returns:
        Any: YAML contents.
    '''
    with open(yaml_path, "r", encoding="utf8") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)

    return yaml_data


def check_config(config):
    '''Validate configuration.

    Args:
        config (dict): Configuration as loaded from YAML.

    Raises:
        ValueError: If required vars are missing from the configuration.

    Returns:
        config (dict): Validated configuration.
    '''
    required_vars = {
        "trainData": "dict of datasets and Kaggle download paths",
        "modelType": "Classic or Quantum"
    }
    optional_vars = {
        "outputsPath": "./results",
        "manualSeed": 999,
        "kagglePath": "/tmp",
        "dataPath": "./data",
        "modelPath": "./models",
        "valSize": 0.2,
        "testSize": 0.0,
        "numWorkers": 2,
        "batchSize": 32,
        "imageHeight": 32,
        "imageWidth": 32,
        "numChannels": 3,
        "numEpochs": 10,
        "learningRate": 0.0002,
        "beta1": 0.5,
        "nGPUs": 0,
        "nCPUs": 4,
        "nQubits": 4,
        "qubitType": "default.qubit",
        "diffType": "best",
        "nLayers": 1,
        "typeLayers": "strong",
        "quanType": "HQNN",
        "trainable": False,
        "patch": False
    }

    # Check if required vars are specified
    for required_var, required_val in required_vars.items():
        if not config.get(required_var, None):
            raise ValueError(f"{required_var} must be specified in the "
                             f"configuration as {required_val}")

    # If optional vars are not specified, set default values
    for optional_var, default_val in optional_vars.items():
        if optional_var not in config.keys():
            config[optional_var] = default_val

        if "Path" in optional_var:
            check_path(config[optional_var])

    if config["testSize"] == 0.0 and "testData" not in config:
        raise ValueError("If no testSize is provided, please provide testData.")

    if config["testSize"] == 1.0:
        raise ValueError("Test dataset must be a subset of validation dataset.")

    if config["modelType"] not in ["Classic", "Quantum"]:
        raise ValueError("Please provide a valid modelType: Classic or Quantum")

    if config["quanType"] not in ["HQNN", "QNN"]:
        raise ValueError("Please provide a valid quanType: HQNN or QNN")

    os.environ["KAGGLEHUB_CACHE"] = config["kagglePath"]

    return config


def check_path(dir_path):
    '''Checks if directory exists at given path and creates it if not.

    Args:
        dir_path (str): Directory path
    '''
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def save_results(data, filepath):
    with open(filepath, "w", encoding="utf8") as out:
        writer = csv.writer(out, lineterminator="\n")
        writer.writerow(["Epoch", "Accuracy", "Loss"])

        for epoch, results in data.items():
            writer.writerow(
                [epoch] +
                ["{:.2f}".format(results["accuracy"])] +
                ["{:.2f}".format(results["loss"])]
            )


def save_plot(title, data, filepath):
    fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(10, 5))
    fig.suptitle(title)

    for model, results in data.items():
        xvals = list(results.keys())
        accuracy = [ep.get("accuracy") for ep in results.values()]
        loss = [ep.get("loss") for ep in results.values()]

        ax1.plot(xvals, accuracy, label=model)
        ax2.plot(xvals, loss, label=model)

        if len(xvals) >= 100:
            ax2.set_xticks(np.arange(0, len(xvals), 5))

    ax1.set(ylabel="avg accuracy (%)")
    ax1.set_title("Average accuracy every training epoch")
    ax1.grid()
    ax1.set_ylim([0, 100])

    ax2.set(xlabel="epoch", ylabel="avg loss")
    ax2.set_title("Average loss every training epoch")
    ax2.grid()

    if len(data) > 1:
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    fig.savefig(filepath, bbox_inches='tight')
