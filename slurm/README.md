# How to run deepfake-detection models with GPUs on Slurm

## Environment setup

Log into Cirrus and clone this repository under your shared project space (e.g., `/work/<PROJECT_ID>/<PROJECT_ID>/shared`).

Load the pytorch module:

```console
module load pytorch/2.2.0-gpu
```

Set up a virtual environment to install software dependencies:

```console
cd /work/<PROJECT_ID>/<PROJECT_ID>/shared/deepfake-detection
python -m venv --system-site-packages deepfake_venv
extend-venv-activate deepfake_venv
source deepfake_venv/bin/activate
python -m pip install -r requirements.txt
```

## Data preparation

Job nodes on Cirrus don't have access to the internet so data downloads must be done on the login node before submitting the job.

```console
cd /work/<PROJECT_ID>/<PROJECT_ID>/shared/deepfake-detection
source deepfake_venv/bin/activate
```

Make sure the datasets you need downloaded are specified in the configuration file before running the following after replacing `CONFIG_PATH` with the path to the configuration file:

```console
python deepfake_detection.py -c CONFIG_FILEPATH
```

Once the datasets are downloaded and sorted, you will be asked if you want to continue
to training. Training should only be run as part of a job and not on the login node, so
this should be cancelled.

## Submit job

Check that the slurm job file runs the correct command and uses the correct configuration file:

```console
cd /work/<PROJECT_ID>/<PROJECT_ID>/shared/deepfake-detection/gpu_enabled/slurm/
cat fake_det_job_script.ll
```

Submit the job:

```console
sbatch fake_det_job_script.ll
```

Check job output files replacing `JOB_ID` with the ID from the output of the `sbatch` command:

```console
tail -f slurm-JOB_ID.out
```

To stop watching the file, run Ctrl+C.

To check running jobs, run the following replacing `USERNAME` with your Cirrus username:

```console
squeue -u USERNAME
```

To cancel/delete a job, run the following replaing `JOB_ID` with the ID from the output of the `sbatch` command:

```console
scancel JOB_ID
```

**Note:** Keep in mind that jobs have a lifetime mentioned at the top of the job file (e.g., `#SBATCH --time=01:00:00`), so jobs will automatically be deleted once this time has elapsed. If you're planning on running a slow model, make sure you increase this. The maximum runtime for the main gpu queue is 4 days. See other queues [here](https://docs.cirrus.ac.uk/user-guide/gpu/#quality-of-service-qos).
