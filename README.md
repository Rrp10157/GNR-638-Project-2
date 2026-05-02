# GNR 638 Project 2 VQA Submission

This repository contains an offline inference pipeline for deep-learning MCQ images.
It reads `test.csv`, loads each image from `images/`, and writes `submission.csv`
in the current project directory.

## Folder Structure

```text
.
+-- inference.py
+-- requirements.txt
+-- environment.yml
+-- models/
|   +-- qwen-vl/              # created by setup.bash
+-- sample_test_project_2/
    +-- images/
    +-- test.csv
+-- sample_submission.csv
```

## Grading Commands

```bash
cd ./your_directory
bash setup.bash
conda activate gnr_project_env
python inference.py --test_dir <absolute_path_to_test_dir>
python <grading_script> --submission_file submission.csv
conda remove --name gnr_project_env --all -y
```

## Model

The default model is `Qwen/Qwen3.5-9B`, downloaded during `setup.bash` into
`models/qwen-vl`. Inference uses `local_files_only=True`, so it does not require
internet access during grading.

For a smaller local dry run on a 6 GB GPU, override the model before setup:

```bash
MODEL_ID=Qwen/Qwen3.5-4B bash setup.bash
```

or force 4-bit loading during inference:

```bash
USE_4BIT=1 python inference.py --test_dir sample_test_project_2
```

## Single File Zip Submission

If only `setup.bash` is submitted in the zip, set this line before zipping:

```bash
REPO_URL="https://github.com/<your-user>/<your-public-repo>.git"
```

The setup script will clone the public repository, create `gnr_project_env` with
Python 3.11, install CUDA 12.6 PyTorch packages, download model weights, and
leave `inference.py` ready in the grading directory.
