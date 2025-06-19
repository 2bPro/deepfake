#!/usr/bin/python3
'''Load data and train model with parameters as specified in the configuration
   file.
'''
import os
import glob
import argparse
import logging
import shutil

import torch

from deepfake_detection import utils, data, devices, train
from deepfake_detection.models import classical, hybrid, quantum


def argparser():
    '''Parses CLI arguments.

    Returns:
       ArgparseNamespace: Dictionary of CLI arguments.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", help="Path to config YAML file.",
                        type=str, required=True)
    parser.add_argument("--offline", "-o", help=("Skip data download."),
                        action=argparse.BooleanOptionalAction)
    parser.add_argument("--resort", "-r", help=("Resorts pre-downloaded data."),
                        action=argparse.BooleanOptionalAction)
    parser.add_argument("--infer", "-i", help=("Path to saved model params."),
                        type=str, required=False)

    return parser.parse_args()


def create_model(config):
    # Create model
    if config["modelType"] == "Classic":
        if args.offline:
            model = classical.ClassicalModel(config["modelPath"])
        else:
            model = classical.ClassicalModel()
    elif config["modelType"] == "Quantum":
        quantum_device = devices.QuantumDevice(
            config["qubitType"], config["nQubits"], config["nLayers"],
            config["typeLayers"], config["diffType"]
        )

        if config["quanType"] == "HQNN":
            model = hybrid.HQNN(quantum_device)
        elif config["quanType"] == "QNN":
            if config["trainable"]:
                if config["patch"]:
                    print("Run TrainablePatchQNN")
                else:
                    print("Run TrainableQNN")
            else:
                model = quantum.QNN(quantum_device)

    return model


def full_run(config, outputs_path, session_name):
    # Configure logging
    utils.setup_log(outputs_path, session_name, "info")
    logging.info("Loaded configuration: %s", config)

    # Set seed for reproducibility
    seed = config["manualSeed"]
    torch.manual_seed(seed)
    logging.info("Seed: %s", seed)

    # Create device
    device = devices.ClassicDevice(config["nGPUs"]).device
    logging.info("Device type: %s", device.type)

    # Create model
    model = create_model(config)

    # Prepare data
    train_path = os.path.join(config["dataPath"], "train")
    test_path = os.path.join(config["dataPath"], "test")

    ## Download and sort only if online
    if not args.offline:
        ## Download and sort data
        data.download(config["trainData"])
        kaggle_path = os.path.join(os.environ["KAGGLEHUB_CACHE"], "datasets")
        data.sort(kaggle_path, train_path, config["trainData"])

        if config.get("testData", None):
            data.download(config["testData"]) # will be cached if same as train
            data.sort(kaggle_path, test_path, config["testData"])

        ## Check if user wants to continue
        input("Press Enter if you want to continue to training or Ctrl+C to stop.")

    if args.resort:
        kaggle_path = os.path.join(os.environ["KAGGLEHUB_CACHE"], "datasets")
        data.sort(kaggle_path, train_path, config["trainData"])

        if config.get("testData", None):
            data.sort(kaggle_path, test_path, config["testData"])

    ## Import data
    dataset = data.load(train_path, config["imageHeight"], config["imageWidth"])

    ## Split data
    train_dataset, valid_dataset = data.split(dataset, config["valSize"])

    test_dataset = ""

    if config["testSize"] and config["testSize"] > 0.0:
        valid_dataset, test_dataset = data.split(valid_dataset, config["testSize"])
    else:
        logging.info("Preparing new dataset for inference.")
        test_dataset = data.load(test_path, config["imageHeight"], config["imageWidth"])

    train_percent = int((1 - config["valSize"])*100)
    valid_percent = int(config["valSize"]*config["testSize"]*100) if config["testSize"] > 0 else int(config["valSize"]*100)
    test_percent = int((config["valSize"] - config["valSize"]*config["testSize"])*100) if config["testSize"] > 0 else 0
    logging.info("Data split train:validate:test is %s:%s:%s percent or %s:%s:%s number of images.", train_percent, valid_percent, test_percent, len(train_dataset), len(valid_dataset), len(test_dataset))

    ## Prepare data
    if config["modelType"] == "Quantum" and config["quanType"] == "QNN" and not config["trainable"]:
        train_dataset = data.preprocess(train_dataset, model.quanv_4, config["nCPUs"])
        valid_dataset = data.preprocess(valid_dataset, model.quanv_4, config["nCPUs"])
        test_dataset = data.preprocess(test_dataset, model.quanv_4, config["nCPUs"])

    train_dataloader = data.create_dataloader(train_dataset, config["batchSize"], config["numWorkers"])
    valid_dataloader = data.create_dataloader(valid_dataset, config["batchSize"], config["numWorkers"])
    test_dataloader = data.create_dataloader(test_dataset, config["batchSize"], config["numWorkers"])

    ## Parallelise data processing
    if device.type == "cuda" and config["nGPUs"] > 1:
        model = torch.nn.DataParallel(model, list(range(config["nGPUs"])))

    logging.info("Type of model: %s", model)

    # Train model
    train_results = train.train(
        device, model, train_dataloader, valid_dataloader,
        config["numEpochs"], config["batchSize"], config["learningRate"]
    )

    # Save results
    results_path = os.path.join(outputs_path, f"{session_name}.csv")
    utils.save_results(train_results, results_path)

    logging.info("Saved training results at %s.", results_path)

    # Save model
    model_path = os.path.join(outputs_path, f"{session_name}.pth")
    torch.save(model.state_dict(), model_path)
    logging.info("Saved trained model at %s.", model_path)

    # Save plot
    utils.save_plot(
        f"{session_name} training results",
        {session_name: train_results},
        os.path.join(outputs_path, f"{session_name}.png")
    )

    # Test model
    train.test(model, device, test_dataloader)


def inference_only(config, model_params, outputs_path, session_name):
    # Configure logging
    datasets = "_".join(config["testData"].keys())
    utils.setup_log(outputs_path, f"{session_name}_test_on_{datasets}", "info")
    logging.info("Loaded configuration: %s", config)

    # Create device
    device = devices.ClassicDevice(config["nGPUs"]).device
    logging.info("Device type: %s", device.type)

    # Redeclare the model and load the saved parameters
    model = create_model(config)
    logging.info("Loading model parameters from %s.", model_params)

    model.load_state_dict(torch.load(model_params))
    model.eval()

    # Prepare data
    test_path = os.path.join(config["dataPath"], "test")

    ## Download and sort only if online
    if not args.offline:
        ## Download data
        data.download(config["testData"])

        ## Sort data
        kaggle_path = os.path.join(os.environ["KAGGLEHUB_CACHE"], "datasets")
        data.sort(kaggle_path, test_path, config["testData"])

        ## Check if user wants to continue
        input("Press Enter if you want to continue to inference or Ctrl+C to stop.")

    if args.resort:
        kaggle_path = os.path.join(os.environ["KAGGLEHUB_CACHE"], "datasets")
        data.sort(kaggle_path, test_path, config["testData"])

    test_dataset = data.load(test_path, config["imageHeight"], config["imageWidth"])

    if config["modelType"] == "Quantum" and config["quanType"] == "QNN" and not config["trainable"]:
        test_dataset = data.preprocess(test_dataset, model.quanv_4, config["nCPUs"])

    test_dataloader = data.create_dataloader(test_dataset, config["batchSize"], config["numWorkers"])

    train.test(model, device, test_dataloader)


def main(args):
    '''Main function loading config, loading data, splitting it into training
       and valing datasets, training and saving a model.

    Args:
        args (ArgparseNamespace): Dictionary of CLI arguments.
    '''
    config = utils.check_config(utils.load_yaml(args.config))

    # Set up session-specific outputs directory
    datasets = "_".join(config["trainData"].keys())
    model_name = config["modelType"]

    if config["modelType"] != "Classic":
        model_name = config["quanType"]

    session_name = f"{model_name}_bs{config['batchSize']}_ep{config['numEpochs']}_{datasets}"

    if torch.cuda.is_available() and config["nGPUs"] > 0:
        session_name += f"_gpu{config['nGPUs']}"

    outputs_path = os.path.join(config["outputsPath"], session_name)
    utils.check_path(outputs_path)

    # Make a copy of the configuration
    try:
        shutil.copyfile(args.config, os.path.join(outputs_path, f"{session_name}.conf.yaml"))
    except shutil.SameFileError:
        pass

    if args.infer:
        inference_only(config, args.infer, outputs_path, session_name)
    else:
        full_run(config, outputs_path, session_name)


if __name__ == '__main__':
    args = argparser()
    main(args)
