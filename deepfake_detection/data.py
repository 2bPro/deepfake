#!/usr/bin/python3
'''Contains all data manipulation modules: download, sort, load.

   Requires a YAML file listing the dataset names (used as folder name), and
   kaggle URLs. For example:

   datasets:
    dataset1: "source/dataset1"
    dataset2: "source/dataset2"

   Kagglehub will automatically download datasets in the user home directory.
   This is not ideal when working in a team and/or on remote systems. By
   default, this script will use ./data_download, but you can also specify a
   location in the YAML file:

   kagglePath: "/new/location"
'''

import os
import shutil
import logging

import kagglehub
import sklearn
import numpy as np
import torch
import torchvision as torchvis
import multiprocessing


def download(datasets):
    '''Downloads datasets listed in configuration from Kaggle.

    Args:
        datasets (dict): Datasets and Kaggle links to download.
    '''
    logging.info("Downloading data")

    for dataset_path in datasets.values():
        kagglehub.dataset_download(dataset_path)


def sort(source_path, dest_path, datasets):
    '''Merge downloaded datasets in one data folder and sort in fake/real
       directories:

       data/
         fake/
         real/

    Args:
        source_path (str): Kaggle download location.
        dest_path (str): Location of merged output dataset.
        datasets (dict): Datasets and Kaggle links to download.
    '''
    logging.info("Sorting data into fake and real directories at %s.", dest_path)

    if os.path.exists(dest_path):
        shutil.rmtree(dest_path)

    os.makedirs(dest_path)
    os.makedirs(os.path.join(dest_path, "fake"))
    os.makedirs(os.path.join(dest_path, "real"))

    for dataset_path in datasets.values():
        dataset_path = os.path.join(source_path, dataset_path)

        for root, _, files in os.walk(dataset_path):
            if files and files[0].endswith(".jpg"):
                img_dir = os.path.basename(root).lower()
                dest_dir = ""

                if any(txt in img_dir.lower() for txt in ["fake", "1"]):
                    dest_dir = os.path.join(dest_path, "fake")

                if any(txt in img_dir.lower() for txt in ["real", "0"]):
                    dest_dir = os.path.join(dest_path, "real")

                for file in files:
                    shutil.copy(os.path.join(root, file), os.path.join(dest_dir, file))


def load(data_src, img_height, img_width):
    '''Loads, standardises, resizes, centeres, normalises and reformatts data
       into a Torch tensor. This dataset is then split into train and test
       datasets and split into batch sizes for training.

    Args:
        data_src (str): Path to image data.
        img_height (int): Spatial height size of training images.
        img_width (int): Spatial width size of training images.

    Returns:
        train_dataset: Torch dataset object of images.
        valid_dataset: Torch dataset object of images.
    '''
    logging.info("Loading data")
    dataset = torchvis.datasets.ImageFolder(
        root=data_src,
        transform=torchvis.transforms.Compose([
            torchvis.transforms.RandomHorizontalFlip(p=0.5),
            torchvis.transforms.RandomVerticalFlip(p=0.5),
            torchvis.transforms.RandomRotation(45),
            torchvis.transforms.Resize((img_height, img_width)),
            torchvis.transforms.CenterCrop((img_height, img_width)),
            torchvis.transforms.ToTensor(),
            torchvis.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    )

    return dataset


def split(dataset, split):
    logging.info("Splitting data")
    dataset1, dataset2 = sklearn.model_selection.train_test_split(
        dataset, test_size=split
    )

    return dataset1, dataset2


def process_img(image, img_process_funct):
    img = np.transpose(np.array(image[0]), (1, 2, 0))
    processed_img = img_process_funct(img)
    processed_img = np.transpose(processed_img, (2, 0, 1))

    post = (
        torch.tensor(processed_img, dtype=torch.float32),
        torch.tensor(image[1], dtype=torch.long)
    )

    return post


def preprocess(data, img_process_funct, cpus):
    logging.info("Preprocessing data")
    logging.info("Total images to process: %s", len(data))

    proc_data = []

    with multiprocessing.Pool(processes=cpus) as pool:
        logging.info(f"Creating {cpus} threads for preprocessing images...")
        async_results = [
            pool.apply_async(process_img, [data[i], img_process_funct])
            for i in range(len(data))
        ]

        for result in async_results:
            try:
                processed_img = result.get()
            except Exception:
                logging.exception("A child process raised an exception")
            else:
                proc_data.append(processed_img)

    return proc_data


def create_dataloader(dataset, batch_size, number_workers):
    '''Split loaded into batch sizes for training.

    Args:
        dataset (obj): Torch dataset object of images.
        batch_size (int): Dataset batch size.
        number_workers (int): Number of dataloader workers.
        validation (bool): If False, dataset type is "Train", else "Valid".
                           Defaults to False.

    Returns:
        train_dataloader: Torch dataloader object of batched data for training.
        test_dataloader: Torch dataloader object of batched data for testing.
    '''
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        shuffle=True, num_workers=number_workers
    )

    return dataloader
