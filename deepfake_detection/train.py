#!/usr/bin/python3
import logging
from timeit import default_timer as timer

import torch


def train(device, model, train_data, val_data, epochs, batch_size, learn_rate):
    '''Train and validate model.

    Args:
        device (obj): torch device
        model (obj): model
        train_data (_type_): torch train dataloader
        val_data (_type_): torch validation dataloader
        epochs (_type_): number of epochs
        batch_size (_type_): batch size during training
        learn_rate (_type_): learning rate for optimiser
    '''
    count = 0
    loss_list = []
    iteration_list = []
    accuracy_list = []
    results = {}
    model = model.to(device)

    # Create optimiser, loss functions, noise, and labels
    optimiser = torch.optim.Adamax(model.parameters(), lr=learn_rate)
    loss = torch.nn.CrossEntropyLoss()

    logging.info("Starting Training Loop...")

    if device.type == "cuda":
        gpu_util = []

    exec_start = timer()

    for epoch in range(epochs):
        running_loss = 0
        for xs, ys in train_data:
            optimiser.zero_grad()
            xs = xs.to(device)
            ys = ys.to(device)

            loss_evaluated = loss(model(xs), ys)
            loss_evaluated.backward()
            optimiser.step()

            if device.type == "cuda":
                gpu_util.append(torch.cuda.utilization(device=None))

            running_loss += loss_evaluated.item()
            count += 1

            correct = 0

            if count % 50 == 0:
                for images, labels in val_data:
                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = model(images)
                    predictions = torch.argmax(outputs, axis=1).cpu().detach().numpy()
                    correct = [1 if p == p_true else 0 for p, p_true in zip(predictions, labels)]

                accuracy = 100 * sum(correct) / len(correct)
                loss_list.append(running_loss)
                iteration_list.append(count)
                accuracy_list.append(accuracy)

                logging.info(f"Iteration: {count}  Loss: {running_loss:.4f}  Accuracy: {accuracy:.2f}%")

        avg_loss = running_loss / batch_size
        avg_accuracy = sum(accuracy_list) / len(accuracy_list) if len(accuracy_list) > 0 else 0

        logging.info(f"Epoch: {epoch + 1}/{epochs}  Loss: {avg_loss:.4f}  Accuracy: {avg_accuracy:.2f}%")
        results[f"ep{epoch + 1}"] = {"accuracy": avg_accuracy, "loss": avg_loss}

    exec_end = timer()
    time_elapsed = exec_end - exec_start
    logging.info(f"Execution time: {time_elapsed:.2f}s")

    if device.type == "cuda":
        avg_gpu_util = sum(gpu_util) / len(gpu_util)
        logging.info(f"Avg GPU utilization: {avg_gpu_util:.2f}%")

    return results

def test(model, device, dataset):
    count = 0
    accuracy_list = []
    model = model.to(device)

    logging.info(f"Starting inference test on {len(dataset.dataset)} images")

    if device.type == "cuda":
        gpu_util = []

    for images, labels in dataset:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        if device.type == "cuda":
            gpu_util.append(torch.cuda.utilization(device=None))

        predictions = torch.argmax(outputs, axis=1).cpu().detach().numpy()
        correct = [1 if p == p_true else 0 for p, p_true in zip(predictions, labels)]

        accuracy = 100 * sum(correct) / len(correct)

        accuracy_list.append(accuracy)
        count += 1

        if count == 100:
            logging.info(f"Iteration: {count}  Accuracy: {accuracy:.2f}%")

    avg_accuracy = sum(accuracy_list) / len(accuracy_list) if len(accuracy_list) > 0 else 0
    logging.info(f"Average testing accuracy: {avg_accuracy:.2f}%")

    if device.type == "cuda":
        avg_gpu_util = sum(gpu_util) / len(gpu_util)
        logging.info(f"Avg GPU utilization: {avg_gpu_util:.2f}%")

    return avg_accuracy
