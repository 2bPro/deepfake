# How to run deepfake-detection models with GPUs on Safe Haven Services

## 1. Prepare data

Log on to `eidf114` machine and clone the code:

```console
$ git clone https://gitlab.eidf.ac.uk/Bianca/deepfake.git
```

Run the code to download and sort the data:

```
$ cd deepfake && python3 -m venv venv && source venv/bin/activate
(venv) $ pip install -r requirements.txt
(venv) $ python3 deepfake_detection.py -c Deepfake.conf.yaml
```

Compress data:

```console
$ tar cf deepfake_data.tar data
```

Ask the Reasearch Coordinator (RC) to generate a ServU link for uploading the data to the Safe Haven.
