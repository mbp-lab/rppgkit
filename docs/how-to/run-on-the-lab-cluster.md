# How to run rppgkit on the lab cluster

This guide shows you how to run a POS evaluation as SLURM jobs on the lab cluster, and
then how to run scripts interactively, change the requested resources and use other
partitions. The partitions of the cluster and the resources each script requests are
listed at the end.

All jobs in this guide are submitted from the login node of the cluster. To run a
script interactively on a compute node instead, see
[Run a script interactively](#run-a-script-interactively).

## Prerequisites

- A connection from within the TechFak network (CITEC WiFi or wired network,
  `shell.techfak.de`, or the VPN), so that you can log in to the login node:

  ```bash
  ssh <user>@login-1.gpu.cit-ec.net
  ```

- [uv](https://docs.astral.sh/uv/) installed in your home directory on the cluster:

  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

- Read access to `/vol/rppg-challenge/`, and write access to
  `/vol/rppg-challenge/processed/` to build caches.

## Set up rppgkit on the cluster

1. On the login node, clone the repository and install the dependencies:

   ```bash
   git clone https://github.com/mbp-lab/rppgkit.git
   cd rppgkit
   uv sync --locked
   ```

2. Check that the configured folders exist on the cluster. The paths are already set
   for the lab cluster:

   | File                             | Key           | Path                             | Content                          |
   | -------------------------------- | ------------- | -------------------------------- | -------------------------------- |
   | `configs/dataset/ubfc_rppg.yaml` | `root`        | `/vol/rppg-challenge/raw/`       | the subject folders of UBFC-rPPG |
   | `configs/paths.yaml`             | `cache_root`  | `/vol/rppg-challenge/processed/` | the caches                       |
   | `configs/paths.yaml`             | `mlflow_root` | `~/rppgkit/mlflow_demo/`         | the MLflow runs                  |

   ```bash
   ls /vol/rppg-challenge/raw/ /vol/rppg-challenge/processed/
   ```

   The first folder contains one folder per subject, for example `subject1/`.

## Run a POS evaluation

A POS evaluation consists of one GPU stage and one CPU stage. Run the following
commands from the repository folder on the login node.

1. Build the ROI cache with YOLO5Face. The job runs on the `gpu` partition:

   ```bash
   uv run python scripts/1_extract_roi.py -m
   ```

2. Create the split file. It only reads the subject list of the dataset, so run it
   without `-m`; it then runs on the login node and does not submit a job:

   ```bash
   uv run python scripts/3_make_splits.py split=test_only
   ```

3. Wait until the job of step 1 has finished. It no longer appears in the list of
   your jobs:

   ```bash
   squeue --me
   ```

4. Evaluate POS. The job runs on the `cpu` partition and requests no GPU:

   ```bash
   uv run python scripts/evaluate.py -m method=pos split=test_only
   ```

The evaluation writes `metrics.json` to `multirun/<date>/<time>/0/`.

## Follow a job

Each job writes its files to `multirun/<date>/<time>/.submitit/<jobid>/`:

- `<jobid>_0_log.out`: the log of the script.
- `<jobid>_0_log.err`: warnings and tracebacks.

To check a job, use the following commands:

```bash
squeue --me                                                          # queued and running jobs
tail -f multirun/<date>/<time>/.submitit/<jobid>/<jobid>_0_log.out   # follow the log
sacct -j <jobid>                                                     # state and exit code
seff <jobid>                                                         # CPU and memory used
```

Do not submit `evaluate.py` with a GPU launcher for POS. The GPU would remain unused
for the whole job and would not be available to other users.

## Build the ROI cache without a GPU

To detect faces with the Haar detector, select the `haar72` launcher together with
the ROI configuration. The job runs on the `cpu` partition:

```bash
uv run python scripts/1_extract_roi.py -m roi=haar72 hydra/launcher=haar72
```

## Run a script interactively

To try a script without submitting it, for example a short training run while you
develop a method, start an interactive shell on a compute node with `srun --pty`. The
shell starts in the folder from which you ran `srun`, so run it from the repository
folder. For a script that uses a GPU, request the same resources as the `gpu` launcher:

```bash
srun --partition=gpu --gres=gpu:a40:1 --cpus-per-task=5 --mem=32G --time=01:00:00 --pty bash
```

For a CPU-only script, request the `cpu` partition and no GPU:

```bash
srun --partition=cpu --cpus-per-task=4 --mem=16G --time=01:00:00 --pty bash
```

When the shell has started on the node, run the script without `-m`, for example a
training run of one epoch:

```bash
uv run python scripts/train.py trainer.max_epochs=1
```

The session holds its resources until you leave the shell with `exit` or the time
limit is reached. Leave the shell as soon as you have finished, so that the GPU is
available to other users.

## Request more CPU cores

SLURM restricts each job to the number of CPU cores that it requests
(`cpus_per_task`). All processes and threads of the script share these cores, however
many cores the node has. To request more cores for a single run, override
`hydra.launcher.cpus_per_task`:

```bash
uv run python scripts/evaluate.py -m method=pos hydra.launcher.cpus_per_task=4
```

To decide whether a job needs more cores, run `seff <jobid>` after it has finished. A
CPU efficiency close to 100 % indicates that the job was limited by its cores; a low
CPU efficiency indicates that additional cores would remain idle.

For `train.py`, request one core per data-loading worker plus one for the main
process. For example, with `trainer.num_workers=8`:

```bash
uv run python scripts/train.py -m trainer.num_workers=8 hydra.launcher.cpus_per_task=9
```

## Use another GPU partition

The GPU launchers request one A40 on the `gpu` partition. To use another GPU
partition, override the partition and the generic resource (GRES) together:

```bash
uv run python scripts/train.py -m hydra.launcher.partition=hopper hydra.launcher.additional_parameters.gres=gpu:<type>:1
```

To list the GRES names of each partition, run:

```bash
sinfo -o "%P %G"
```

The GPUs of the `study` partition have 11 GB or 16 GB of memory. A job that fits on
an A40 (46 GB) may therefore run out of GPU memory on these nodes.

## Receive an email when a job ends

Every job sends an email to your TechFak account when it ends or fails. A run with
`-m` is submitted as one job array, which sends one email for the whole array rather
than one per job. To submit a run without emails, override the mail type:

```bash
uv run python scripts/train.py -m hydra.launcher.additional_parameters.mail_type=NONE
```

To change the emails for every run, edit `mail_type` in
`configs/hydra/launcher/slurm_cpu.yaml`.

## Change the default resources

To change the resources for every run, edit the launcher file in
`configs/hydra/launcher/`. The files `yolo5face.yaml`, `haar72.yaml` and `gpu.yaml`
extend `slurm_cpu.yaml` and override only the values that differ.

## Resources requested by each script

Only face detection with YOLO5Face and training use a GPU. The Haar detector,
preprocessing and evaluation of methods without training, such as POS, run on the CPU
alone. With `-m`, each script requests the resources of the launcher file named in its
configuration, in `configs/hydra/launcher/`:

| Script                                   | Hardware | Launcher    | Partition | GPUs    | CPU cores | Memory | Time limit |
| ---------------------------------------- | -------- | ----------- | --------- | ------- | --------- | ------ | ---------- |
| `1_extract_roi.py` with `roi=yolo5face*` | GPU      | `yolo5face` | `gpu`     | 1 × A40 | 4         | 16 GB  | 120 min    |
| `1_extract_roi.py` with `roi=haar72`     | CPU      | `haar72`    | `cpu`     | none    | 4         | 16 GB  | 120 min    |
| `2_preprocess.py`                        | CPU      | `slurm_cpu` | `cpu`     | none    | 1         | 4 GB   | 60 min     |
| `3_make_splits.py`                       | CPU      | none        | none      | none    | –         | –      | –          |
| `evaluate.py`                            | CPU      | `slurm_cpu` | `cpu`     | none    | 1         | 4 GB   | 60 min     |
| `train.py`                               | GPU      | `gpu`       | `gpu`     | 1 × A40 | 5         | 32 GB  | 240 min    |

## Partitions of the lab cluster

| Partition | Nodes             | GPUs per node   | GPU memory | CPU cores per node | RAM per node |
| --------- | ----------------- | --------------- | ---------- | ------------------ | ------------ |
| `cpu`     | `assistant-[1-4]` | none            | –          | 64                 | 443 GB       |
| `study`   | `servant-[1,2]`   | 2 × Tesla P100  | 16 GB      | 2 × 20             | 216 GB       |
| `study`   | `servant-[4,5]`   | 2 × GTX 1080 Ti | 11 GB      | 2 × 20             | 108 GB       |
| `gpu`     | `worker-[1-11]`   | 4 × A40         | 46 GB      | 120                | 443 GB       |
| `hopper`  | `worker-[12-15]`  | 4 × H200        | 140 GB     | 180                | 672 GB       |

To list the partitions and their current state, run `sinfo -s`.
