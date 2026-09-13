# RPS Project

A TensorFlow-based project focused on experimenting with Rock-Paper-Scissors image classification using CNN models and TensorFlow Datasets.

## Overview

This repository is designed as a clean starting point for learning and prototyping computer vision workflows. It includes a basic notebook for loading and inspecting the dataset, a lightweight project structure, and a reproducible dependency setup for local experimentation.

## Features

- Rock-Paper-Scissors dataset loading with TensorFlow Datasets
- Notebook-based experimentation for model development
- Minimal project layout for source code, notebooks, and outputs
- Dependency management through `requirements.txt`

## Project Structure

```text
RPS-Project/
├── app/                  # Application-level code or entry points
├── src/                  # Reusable source code
├── data/                 # Local datasets and prepared data files
├── experiments/          # Experiment outputs, logs, and artifacts
├── models/               # Saved model weights or checkpoints
├── notebooks/            # Jupyter notebooks for exploration and analysis
├── .gitignore            # Git ignore rules
├── README.md             # Project documentation
├── requirements.txt      # Python dependencies
└── .venv/                # Local virtual environment (ignored by Git)
```

## Getting Started

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd RPS-Project
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```cmd
.venv\Scripts\activate.bat
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

## Notebook Usage

Open the notebook in `notebooks/rps_cnn_battle.ipynb` to:

- load the Rock-Paper-Scissors dataset
- inspect dataset metadata
- verify installed TensorFlow and TFDS versions
- continue building the CNN pipeline

## Notes

- Keep large datasets, trained models, and generated artifacts out of GitHub.
- Use the folders under `data/`, `experiments/`, and `models/` for local-only outputs.
- This repository is intended to be a clean base for experimentation and future extension.

## Requirements

- Python 3.11+
- TensorFlow 2.15+
- TensorFlow Datasets
- NumPy
- Matplotlib
- Pandas
- scikit-learn

## License

This project is provided for educational and experimental purposes.
