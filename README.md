# deepfake-detection

## ToC

- [deepfake-detection](#deepfake-detection)
  - [ToC](#toc)
  - [Background](#background)
    - [Objective](#objective)
    - [Metrics](#metrics)
    - [Research](#research)
    - [Model choice](#model-choice)
    - [Data choice](#data-choice)
  - [Environment](#environment)
    - [Local](#local)
    - [Docker](#docker)
      - [Push image to GitLab Container Registry](#push-image-to-gitlab-container-registry)
      - [Push image to EIDF Registry (Harbor)](#push-image-to-eidf-registry-harbor)
    - [GPU services](#gpu-services)
  - [Configuration](#configuration)
  - [Usage](#usage)
  - [Outputs](#outputs)
  - [Compare models](#compare-models)

## Background

### Objective

The main aim of the project is to create a classifier capable of detecting whether an image is real or fake (i.e., AI generated) while showing the difference in performance between classic vs quantum Machine Learning.  

The approach requires the creation and benchmarking of a classic classification model capable of detecting deepfakes followed by the addition of quantum layers to create a hybrid model and its benchmarking for comparison.

### Metrics

When benchmarking the models, a set of metrics is selected to ensure fair comparison:

* Prediction accuracy
* Training loss

For scalability benchmarking, the following metrics will be considered:

* Type of optimiser
* Number of training epochs
* Training batch size
* Resource usage (e.g., GPU usage)

Specifically for quantum benchmarking:

* Number of qubits
* Number of gates
* Circuit depth

Optionally, if time allows, make use of explainable methods to collect information about the features used in the learning process for comparison.

### Research

* [Join the High Accuracy Club on ImageNet with A Binary Neural Network Ticket](https://arxiv.org/abs/2211.12933)
* [Faster Than Lies: Real-time Deepfake Detection using Binary Neural Networks](https://arxiv.org/abs/2406.04932)
* [CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images](https://arxiv.org/abs/2303.14126#)
* [Quantum-Trained Convolutional Neural Network for Deepfake Audio Detection](https://arxiv.org/html/2410.09250v1)
* [Subspace Preserving Quantum Convolutional Neural Network Architectures](https://arxiv.org/abs/2409.18918)
* [Hybrid quantum image classification and federated learning for hepatic steatosis diagnosis](https://arxiv.org/abs/2311.02402)
* [Quantum machine learning for image classification](https://arxiv.org/abs/2304.09224)
* [A Novel Quantum Neural Network Approach to Combating Fake Reviews](https://d1wqtxts1xzle7.cloudfront.net/114003957/s44227_024_00028_x-libre.pdf?1714538109=&response-content-disposition=inline%3B+filename%3DA_Novel_Quantum_Neural_Network_Approach.pdf&Expires=1730373564&Signature=GdYaC9wub7QV6P~81xb9AMdYeNl3Hsk1tqh4CoajaJHeJya~PNGdTNV00yeACXuJW1epx0REB73K0VlP8YnXUjBeBybBpl49N-F4zvTkwgwqGc0fMg7WK1JanuD6J6wIMdRI5RVy39GUVdwOPHjHE-tmWXeoQ1owc2NP1y3SHQoA-z1Y0wkWaL93uh~LB45vD9Fc~3ILkib9FGnMiIasuEYtY6A9OmZIxi00SLVlvEz31s1qj8FZgeeKZ5HZBo0seIVWtyofcnkDH-JF1nYckW9L3VTFP46ju8JZpgjiT07pD8uHWpVzR6SL8oucH2CUQmGyCo3bhcDMOwbIPwPb1A__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA)
* [Pooling techniques in hybrid quantum-classical convolutional neural networks](https://arxiv.org/pdf/2305.05603)

### Model choice

The problem requires a classification model, and due to the fact that we need to classify whether an image is real or fake, this fits very well with Binary Neural Networks.  

There are a few options that were considered for achieving a hybrid model:

* Training a hybrid model from scratch and comparing it with an off-the-shelf model - the concern with this approach is that off-the-shelf models are trained on very large datasets over many resources and hours, which would make for an unfair comparison
* Training a classical model from scratch and adding quantum layers to it - this takes time and resources to train both models but would be a fairer comparison
* Using a pre-trained classical model and adding quantum layers to it - this would reduce the time to train as we could focus on the hybrid model, but may not be as easy to add quantum layers to it

### Data choice

The focus will initially be on facial deepfake detection as there are many open-source datasets available for this use case. Ideally, we would like to expand this to include other types of images.

| Dataset | Kaggle handle |
| ------- | ------------- |
| [hard1k](https://www.kaggle.com/datasets/hamzaboulahia/hardfakevsrealfaces)  | "hamzaboulahia/hardfakevsrealfaces" |
| [deepfake2k](https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection) | "ciplab/real-and-fake-face-detection" |
| [rvf10k](https://www.kaggle.com/datasets/sachchitkunichetty/rvf10k) | "sachchitkunichetty/rvf10k" |
| [140k](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) | "xhlulu/140k-real-and-fake-faces" |

Multiple datasets are sorted, and optionally combined, into `fake` and `real` classes.

## Environment

### Local

To run the code locally, clone the repository and install the requirements in a virtual environment:

```console
cd deepfake-detection
python3 -m venv deepfake_venv
source deepfake_venv/bin/activate
pip install -r requirements.txt
```

### Docker

Build the image and tag it with the version:

```console
docker build -t deepfake:v<X> .
```

To test the image locally, run the container with the three expected mounts and no network:

```console
docker run -v .:/safe_data/deepfake
           -v ./results:/safe_outputs
           -v ./scratch:/scratch
           deepfake:v<X>
```

>**Note:** To test in offline mode (e.g., for simulating a Safe Haven environment), add `--network none` to the docker run command.

### Push image to GitLab Container Registry

Define pipeline:

```yaml
stages:
  - lint
  - build
  - scan

dockerfile-lint:
  stage: lint
  image: hadolint/hadolint:v2.12.0-debian
  script:
    - hadolint --failure-threshold error Dockerfile
  only:
    - tags
    - branches

build-and-push:
  stage: build
  image:
    name: gcr.io/kaniko-project/executor:v1.22.0-debug
    entrypoint: [""]
  script:
    - echo "{\"auths\":{\"$CI_REGISTRY\":{\"username\":\"$CI_REGISTRY_USER\",\"password\":\"$CI_REGISTRY_PASSWORD\"}}}" > /kaniko/.docker/config.json
    - /kaniko/executor
        --context $CI_PROJECT_DIR
        --dockerfile $CI_PROJECT_DIR/Dockerfile
        --destination $CI_REGISTRY_IMAGE:$CI_COMMIT_TAG
  only:
    - tags

trivy-sbom:
  stage: scan
  image:
    name: aquasec/trivy:0.49.1
    entrypoint: [""]
  before_script:
    - mkdir -p ~/.docker
    - echo "{\"auths\":{\"$CI_REGISTRY\":{\"username\":\"$CI_REGISTRY_USER\",\"password\":\"$CI_REGISTRY_PASSWORD\"}}}" > ~/.docker/config.json
  script:
    - trivy image --scanners vuln --format json --output trivy.sbom.json $CI_REGISTRY_IMAGE:$CI_COMMIT_TAG
  artifacts:
    paths:
      - trivy.sbom.json
  only:
    - tags
```

Create variables:

- CI_REGISTRY_PASSWORD (masked and hidden) = GitLab access token with read_registry, write_registry access

### Push image to EIDF Registry (Harbor)

Harbor has integrated Trivy scanning so this doesn't have to be added to the pipeline:

```yaml
stages:
  - lint
  - build

dockerfile-lint:
  stage: lint
  image: hadolint/hadolint:v2.12.0-debian
  script:
    - hadolint --failure-threshold error Dockerfile
  only:
    - tags
    - branches

build-and-push:
  stage: build
  image:
    name: gcr.io/kaniko-project/executor:v1.22.0-debug
    entrypoint: [""]
  script:
    - echo "{\"auths\":{\"$HARBOR_REGISTRY\":{\"username\":\"$HARBOR_USER\",\"password\":\"$HARBOR_PASSWORD\"}}}" > /kaniko/.docker/config.json
    - /kaniko/executor
        --context $CI_PROJECT_DIR
        --dockerfile $CI_PROJECT_DIR/Dockerfile
        --destination $HARBOR_REGISTRY/$HARBOR_PROJECT/$CI_REGISTRY_IMAGE:$CI_COMMIT_TAG
  only:
    - tags
```

Create variables:

- HARBOR_REGISTRY = registry.eidf.ac.uk
- HARBOR_PROJECT = name of the project on Harbor
- HARBOR_USER = Harbor username
- HARBOR_PASSWORD (masked and hidden) = Harbor CLI secret

>**IMPORTANT:** You must check the vulnerability report of your image and try to solve any solvable issues. For Safe Havens, images should aim to not have any high vulnerabilities, however, it is known that this may not be possible with certain base images.

### GPU services

- [EIDF](./kueue/eidf/README.md)
- [SHS](./kueue/shs/README.md)
- [Cirrus](./slurm/README.md)

## Configuration

Both the notebook and script require a configuration file such as [Deepfake.conf.yaml](./Deepfake.conf.yaml), but you can define your own. Here are the recognised parameters:

| Parameter | Required | Description | Format | Default |
| --------- | -------- | ----------- | ------ | ------- |
| `outputsPath` | No | Location for training or inference outputs such as logs, models, and training results. | str | `./results` |
| `manualSeed` | No | Used for training reproducibility. | int between 1 and 10,000 | `999` |
| `modelType` | **Yes** | Type of model to train or test. | One of `Classic` or `Quantum` | |
| `quanType` | No | Type of model. | One of `HQNN` or `QNN`. | `HQNN` |
| `kagglePath` | No | Location for Kaggle downloads. | str | `/tmp` |
| `dataPath` | No | Location for training or testing data. | str | `./data` |
| `modelPath` | No | Location for offline models. | str | `./models` |
| `trainData` | **Yes** | Dataset(s) to be downloaded for training. | [dataset_name: "Kaggle handle"] |  |
| `testData` | only if `testSize` is not specified or is `0.0` | Dataset(s) to be downloaded for testing if subset of `trainData` is not used for testing. | [dataset_name: "Kaggle handle"] |  |
| `valSize` | No | Percent of train data to use for validation during training. | float | `0.2` |
| `testSize` | No | Percent of validation data to use for testing. If not specified or `0.0`, testing will be performed on `testData`. | `0.0` |
| `numWorkers` | No | Number of workers for dataloader(s). | int | `2` |
| `batchSize` | No | Batch size during training. | int | `32` |
| `imageHeight` | Spatial height size of training images. All images will be resized to this size using a transformer. | int | `32` |
| `imageWidth` | No | Spatial width size of training images. All images will be resized to this size using a transformer. | int | `32` |
| `numChannels` | No | Number of channels in the training images. For color images this is 3. | int - 2 for B&W, 3 for RGB | `3` |
| `numEpochs` | No | Number of training epochs. | int | `10` |
| `learningRate` | No | Learning rate for optimizers. | float | `0.0002` |
| `beta1` | No | Beta1 hyperparameter for Adam optimizers. | float | `0.5` |
| `nGPUs` | No | Number of GPUs to be used of the available. | int | `0` |
| `nCPUs` | No | Number of CPUs to be used for image preprocessing. | int | `4` |
| `nQubits` | No | Number of Qubits for Quantum device. | int | `4` |
| `qubitType` | No | Type of Quantum device. | One of `default.qubit`, `lightning.gpu`, `softwareq.qpp`, `nvidia.custatevec`, `nvidia.cutensornet` | `default.qubit` |
| `diffType` | No | Diff method for training. Only used by Quantum models. | One of `backprop`, `adjoint`, `parameter-shift`, `default`, `best` | `best` |
| `nLayers` | No | Number of layers (1 if random). Only used by Quantum models. | int | `1` |
| `typeLayers` | No | Type of layers. Only used by Quantum models. | One of `basic` or `strong`. | `strong` |
| `trainable` | No | Specific to `QNN` models. Defines whether to use trainable weights. | bool | `False` |
| `patch` | No | Specific to `QNN` models. Defines whether to use patched method for trainable QNN. | bool | `False` |

If you want to keep a parameter's default value, you can omit it from your configuration. For example, the following configuration is valid:

```yaml
trainData:
  hard1k: "hamzaboulahia/hardfakevsrealfaces/versions/1"
valSize: 0.2
testSize: 0.5
modelType: "Classic"
```

For more examples, check the [Results](./results) folder.

## Usage

You can run the code via [Jupyter Notebook](DeepfakeDetection.ipynb), or via the [Python script](deepfake_detection.py).

If running as a script, you can check the help documentation by typing:

```console
python deepfake_detection.py -h
```

You can run all the code including data download and sorting, training, and testing by just passing the `-c` or `--config` flag:

```console
python deepfake_detection.py -c Deepfake.conf.yaml
```

Data download and sorting are expensive tasks that are not required to run every time you train or test a model. If you want to skip downloading and sorting the data, you can run it in `-o` or `--offline` mode:

```console
python deepfake_detection.py -c Deepfake.conf.yaml -o
```

Similarly, you may have pre-downloaded data that you want to re-sort before starting any tasks. You can achieve this by specifying the `-r` or `--resort` flag:

```console
python deepfake_detection.py -c Deepfake.conf.yaml -os
```

If you just want to test a pretrained model, use the `-i` or `--infer` flag followed by the path to the saved model parameters.  

Note that the model configuration used for inference must match that used for training because a new model is created and then loaded with the saved parameters, for this you can use the session-specific copy of the configuration:

```console
python deepfake_detection.py -c results/Classic_bs32_ep5_hard1k/Classic_bs32_ep5_hard1k.conf.yaml \
                             -i results/Classic_bs32_ep5_hard1k/Classic_bs32_ep5_hard1k.pth
```

You can also combine flags, for example the following will run inference offline, skipping the download and sorting of data:

```console
python deepfake_detection.py -c results/Classic_bs32_ep5_hard1k/Classic_bs32_ep5_hard1k.conf.yaml \
                             -oi results/Classic_bs32_ep5_hard1k/Classic_bs32_ep5_hard1k.pth
```

And the following will re-sort a pre-downloaded dataset before inference tasks:

```console
python deepfake_detection.py -c results/Classic_bs32_ep5_hard1k/Classic_bs32_ep5_hard1k.conf.yaml \
                             -ori results/Classic_bs32_ep5_hard1k/Classic_bs32_ep5_hard1k.pth
```

## Outputs

The deepfake detection program will create a session-specific folder with all the outputs at the configured `outputsPath`. Each session will contain the following outputs:

* Copy of the configuration file
* Log
* Trained model parameters
* CSV file of average accuracy and loss for every epoch
* Plot of average accuracy and loss

For an example of this, check the [Results](./results) folder.

## Compare models

You may want to visually compare the training accuracy and loss of multiple models. You can do so by using the [plot_results](./plot_results.py) script with a list of the result files you want to include, the plot name, and where it should be saved. For example to create a plot comparing the training results of a Classical, HQNN, and QNN models on the `hard1k` dataset for 5 epochs:

```console
python plot_results.py -r results/Classic_bs32_ep5_hard1k/Classic_bs32_ep5_hard1k.csv \
                          results/HQNN_bs32_ep5_hard1k/HQNN_bs32_ep5_hard1k.csv \
                          results/HQNN_bs32_ep5_hard1k/HQNN_bs32_ep5_hard1k.csv \
                       -t "Classic vs HQNN vs on hard1k over 5 epochs" \
                       -p "results/Classic_HQNN_QNN_bs32_ep5_hard1k.png"
```

This will result in a graph like [this one](./results/Classic_HQNN_QNN_bs32_ep5_hard1k.png).
