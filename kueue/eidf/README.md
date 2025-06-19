# How to run deepfake-detection models with GPUs on EIDF

### 1. Prepare data

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

### 2. Create PVC

```yaml
# deepfake_pvc.yaml
kind: PersistentVolumeClaim
apiVersion: v1
metadata:
 name: deepfake-pvc
spec:
 accessModes:
  - ReadWriteMany
 resources:
  requests:
   storage: 100Gi
 storageClassName: csi-cephfs-sc
```

```console
$ kubectl -n eidf114ns create -f deepfake_pvc.yaml
$ kubectl -n eidf114ns get pvc
```

### 3. Populate PVC

```yaml
# deepfake_lightweight_pvc_job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  generateName: deepfake-lightweight-
  labels:
    kueue.x-k8s.io/queue-name: eidf114ns-user-queue
spec:
  completions: 1
  template:
    metadata:
      name: deepfake-lightweight
    spec:
      containers:
        - name: data-loader
          image: ubuntu:noble-20250127
          command: ["/bin/sh"]
          args:
            - -c
            - >-
              apt update && apt install -y git vim && sleep infinity
          resources:
            requests:
              cpu: 1
              memory: '1Gi'
            limits:
              cpu: 1
              memory: '1Gi'
          volumeMounts:
            - mountPath: /safe_data
              name: volume
      restartPolicy: Never
      volumes:
        - name: volume
          persistentVolumeClaim:
            claimName: deepfake-pvc
```

```console
$ kubectl -n eidf114ns create -f deepfake_lightweight_pvc_job.yaml 
job.batch/deepfake-lightweight-z9cdp created
$ kubectl -n eidf114ns get pods
NAME                               READY   STATUS    RESTARTS   AGE
deepfake-lightweight-z9cdp-dgngx   1/1     Running   0          30s
```

Copy the data to the PVC via the interactive job:

```console
$ kubectl -n eidf114ns cp deepfake_data.tar deepfake-lightweight-z9cdp-dgngx:/safe_data
```

Open an interactive terminal to decompress data:

```console
$ kubectl -n eidf114ns exec --stdin --tty deepfake-lightweight-z9cdp-dgngx -- /bin/sh
```

Clone the repository to easily take outputs or changes to configuration files off the PVC by pushing changes:

```console
# cd /safe_data && git clone https://gitlab.eidf.ac.uk/Bianca/deepfake.git
```

Decompress transferred data:

```
# tar -xvf deepfake_data.tar -C deepfake && rm -rf deepfake_data.tar && ls deepfake
Deepfake.conf.yaml  Dockerfile  README.md  data  deepfake_detection  deepfake_detection.py  kueue  requirements.txt  results  slurm
```

Edit `Deepfake.conf.yaml` to add paths to data and models:

```yaml
dataPath: "/safe_data/deepfake/data"
outputsPath: "/safe_data/deepfake/results"
modelPath: "/deepfake/models/resnet18-f37072fd.pth"
...
```

### 4. Create a secret for EIDF registry to be used by the job

In a second terminal window, create a file for Harbor token to avoid writing it into the command:

```txt
# eidf_registry.env
HARBOR_USERNAME="<harbor_username>"
HARBOR_CLI_SECRET="<harbor_cli_secret>"
```

```console
$ source eidf_registry.env
$ kubectl -n eidf114ns create secret docker-registry harbor-secret \
       --docker-server=registry.eidf.ac.uk \
       --docker-username=$HARBOR_USERNAME \
       --docker-password=$HARBOR_CLI_SECRET
secret/harbor-secret created
```

### 5. Create job

```yaml
# deepfake_job_harbor.yaml
apiVersion: batch/v1
kind: Job
metadata:
  generateName: deepfake-pytorch-
  labels:
    kueue.x-k8s.io/queue-name: eidf114ns-user-queue
spec:
  completions: 1
  template:
    metadata:
      name: deepfake-pytorch
    spec:
      restartPolicy: Never
      imagePullSecrets:
      - name: harbor-secret
      containers:
      - name: deepfake
        image: registry.eidf.ac.uk/eidf114/deepfake:v1.1
        volumeMounts:
          - mountPath: /safe_data
            name: volume
          - mountPath: /dev/shm
            name: dshm
        resources:
          requests:
            cpu: 8
            memory: '32Gi'
          limits:
            cpu: 16
            memory: '64Gi'
            nvidia.com/gpu: 1
      nodeSelector:
        nvidia.com/gpu.product: NVIDIA-A100-SXM4-40GB
      volumes:
        - name: volume
          persistentVolumeClaim:
            claimName: deepfake-pvc
        - name: dshm
          emptyDir:
            medium: Memory
```

```console
$ kubectl -n eidf114ns create -f deepfake_job_harbor.yaml
job.batch/deepfake-pytorch-r8gkk created
$ kubectl -n eidf114ns get pods
NAME                               READY   STATUS             RESTARTS   AGE
deepfake-lightweight-z9cdp-dgngx   1/1     Running            0          16m
deepfake-pytorch-r8gkk-vkz6z       0/1     ContainerCreating  0          21s
$ kubectl -n eidf114ns logs -f deepfake-pytorch-r8gkk-vkz6z
...
```

### 6. Check outputs

In the interactive job terminal window, check the outputs folder:

```console
$ ls /safe_data/deepfake/results
Classic_bs8_ep20_deepfake2k_gpu1
$ ls /safe_data/deepfake/results/Classic_bs8_ep20_deepfake2k_gpu1
Classic_bs8_ep20_deepfake2k_gpu1.conf.yaml
Classic_bs8_ep20_deepfake2k_gpu1.csv
Classic_bs8_ep20_deepfake2k_gpu1.log
Classic_bs8_ep20_deepfake2k_gpu1.png
Classic_bs8_ep20_deepfake2k_gpu1.pth
```

### 7. Clean up

Exit interactive job terminal and remove jobs:

```console
$ kubectl -n eidf114ns get jobs
NAME                         STATUS     COMPLETIONS   DURATION   AGE
deepfake-lightweight-z9cdp   Running    0/1           70m        70m
deepfake-pytorch-r8gkk       Complete   1/1           3m14s      17m
$ kubectl -n eidf114ns delete job deepfake-pytorch-r8gkk
$ kubectl -n eidf114ns delete job deepfake-lightweight-z9cdp
$ kubectl -n eidf114ns get jobs
No resources found in eidf114ns namespace.
```
