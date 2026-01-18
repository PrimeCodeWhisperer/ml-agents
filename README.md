# Unity ML-Agents Training Project

A machine learning project using Unity ML-Agents for training reinforcement learning models with automated data collection and analysis.

## Table of Contents

- [Unity ML-Agents Training Project](#unity-ml-agents-training-project)
  - [Table of Contents](#table-of-contents)
  - [Prerequisites](#prerequisites)
  - [Setup Instructions](#setup-instructions)
    - [1. Install Unity](#1-install-unity)
    - [2. Clone Repository](#2-clone-repository)
    - [3. Unity Setup](#3-unity-setup)
    - [4. Python Environment Setup](#4-python-environment-setup)
      - [Option A: Conda Environment (Recommended)](#option-a-conda-environment-recommended)
      - [Option B: Virtual Environment (venv)](#option-b-virtual-environment-venv)
    - [5. Environment Configuration](#5-environment-configuration)
    - [6. Google Sheets Integration (Optional)](#6-google-sheets-integration-optional)
  - [Usage](#usage)
    - [Running a Single Training Session](#running-a-single-training-session)
    - [Running Multiple Training Sessions](#running-multiple-training-sessions)
    - [Monitoring Training with TensorBoard](#monitoring-training-with-tensorboard)
  - [Running Learning Algorithms](#running-learning-algorithms)
    - [Random Forest Analysis](#random-forest-analysis)
      - [Basic Usage](#basic-usage)
      - [Command-Line Arguments](#command-line-arguments)
      - [Examples](#examples)
      - [Output](#output)
    - [Linear Regression Analysis](#linear-regression-analysis)
      - [Basic Usage](#basic-usage-1)
      - [Command-Line Arguments](#command-line-arguments-1)
      - [Examples](#examples-1)
      - [Output](#output-1)
    - [Choosing Between Models](#choosing-between-models)
    - [Prerequisites](#prerequisites-1)
  - [Data Labels](#data-labels)
    - [Training Run Metadata](#training-run-metadata)
    - [Performance Metrics](#performance-metrics)
    - [Neural Network Metrics](#neural-network-metrics)
    - [System Information](#system-information)
  - [Daily Workflow](#daily-workflow)
    - [Starting Your Work Session](#starting-your-work-session)
    - [Ending Your Work Session](#ending-your-work-session)
    - [Disable Conda Auto-activation (Optional)](#disable-conda-auto-activation-optional)
  - [Troubleshooting](#troubleshooting)
    - [Apple Silicon grpcio Issues (M1/M2/M3/M4 Macs)](#apple-silicon-grpcio-issues-m1m2m3m4-macs)
    - [Windows grpcio Installation Issues](#windows-grpcio-installation-issues)
    - [Python Version Issues](#python-version-issues)
    - [Virtual Environment Activation Issues](#virtual-environment-activation-issues)
    - [Conda Environment Not Found](#conda-environment-not-found)
    - [ML-Agents Command Not Found](#ml-agents-command-not-found)
    - [Dependency Conflicts](#dependency-conflicts)
    - [Import Errors](#import-errors)
    - [Windows PATH Issues](#windows-path-issues)
  - [Project Structure](#project-structure)
  - [Additional Resources](#additional-resources)
  - [License](#license)

## Prerequisites

- Unity Hub and Unity Editor
- Python 3.10.11 or 3.10.12
- Miniconda (recommended) or Python venv
- VS Code or any code editor
- Git

## Setup Instructions

### 1. Install Unity

Download and install Unity Hub from [https://unity.com/download](https://unity.com/download)

### 2. Clone Repository

Clone this repository to your local machine:

```bash
cd <repositories-path>  # or your preferred location
git clone https://github.com/PrimeCodeWhisperer/ml-agents
cd ml-agents
```


### 3. Unity Setup

1. Launch Unity Hub and sign in with your Unity ID (or create a new account)
2. In Unity Hub, click **Add** → **Add project from disk**
3. Navigate to the cloned repository and select the folder named `Project`
4. Open the project in Unity

### 4. Python Environment Setup

#### Option A: Conda Environment (Recommended)

**Install Miniconda:**

**macOS:**
```bash
brew install --cask miniconda
~/miniconda3/bin/conda init zsh  # or bash if using bash
exec $SHELL -l  # restart shell
conda --version  # verify installation
```

**Windows:**
1. Download Miniconda installer from [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
2. Run the installer (choose "Add Miniconda3 to PATH" during installation)
3. Open **Anaconda Prompt** (or restart your terminal)
4. Verify installation:
```bash
conda --version
```

**Linux:**
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
conda --version  # verify installation
```

**Create and activate environment:**

**macOS/Linux:**
```bash
cd ~/Documents/Uni_Projects/ml-agents  # your project directory
conda create -n mlagents python=3.10.12 -y
conda activate mlagents
```

**Windows (Anaconda Prompt or PowerShell):**
```bash
cd C:\Users\YourUsername\Documents\Uni_Projects\ml-agents
conda create -n mlagents python=3.10.12 -y
conda activate mlagents
```

**Install ML-Agents from local repositories (editable mode):**
```bash
pip install --upgrade pip setuptools wheel
pip install -e ./ml-agents-envs
pip install -e ./ml-agents
```

**Install TensorBoard:**
```bash
pip install tensorboard
```

**Install project-specific dependencies:**
```bash
python -m pip install gspread oauth2client pandas python-dotenv pyyaml
```

**Verify installation:**
```bash
mlagents-learn --help
python -c "import mlagents; print('✅ ML-Agents ready')"
```

#### Option B: Virtual Environment (venv)

**Create and activate virtual environment:**

```bash
# Verify Python version
python -V  # should show Python 3.10.11 or 3.10.12

# Create virtual environment
python -m venv venv

# Activate (Windows - Command Prompt)
venv\Scripts\activate.bat

# Activate (Windows - PowerShell)
venv\Scripts\Activate.ps1

# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install --upgrade pip setuptools wheel
pip install -e ./ml-agents-envs
pip install -e ./ml-agents
pip install tensorboard
python -m pip install gspread oauth2client pandas python-dotenv pyyaml
```

### 5. Environment Configuration

Create a `.env` file in the project root with the following variables:

**macOS/Linux:**
```
UNITY_ENV_PATH=/path/to/unity/build
CONFIG_PATH=/path/to/config
TRAINING_DATA_PATH=/path/to/training/data
```

**Windows:**
```
UNITY_ENV_PATH=C:\path\to\unity\build
CONFIG_PATH=C:\path\to\config
TRAINING_DATA_PATH=C:\path\to\training\data
```

### 6. Google Sheets Integration (Optional)

To automatically log training results to Google Sheets:

1. Create a `credentials.json` file containing your Google Sheets API credentials
2. Place it in the project root directory
3. Add the path to your global CSV file in the `.env` file

## Usage

### Running a Single Training Session

To train the 3DBall example:

```bash
# Make sure your environment is activated
conda activate mlagents  # or: source venv/bin/activate (macOS/Linux)
                        # or: venv\Scripts\activate (Windows)

# Run training
mlagents-learn config/ppo/3DBall.yaml --run-id=3DBallRun1
```

### Running Multiple Training Sessions

For automated batch training:

```bash
python run_training.py [number_of_runs]
```

**Example:**
```bash
python run_training.py 5  # runs 5 training sessions
```

**Important:** This requires the Unity build folder in your top directory:
- **macOS:** Use the `.app` build
- **Windows:** Use the `.exe` build

Results will automatically be added to the Google Sheets CSV file if credentials are configured.

### Monitoring Training with TensorBoard

1. Start TensorBoard:
```bash
tensorboard --logdir results
```

2. Open the returned URL in your browser (typically `http://localhost:6006`)
3. Select the specific training run you want to analyze

## Running Learning Algorithms

After collecting training data through multiple runs, analyze the results using machine learning models to predict performance metrics and understand feature importance.

### Random Forest Analysis

The Random Forest trainer supports three cross-validation strategies and multiple configurable parameters:

#### Basic Usage

```bash
python random_forest_trainer.py
```

This runs with default settings: K-Fold cross-validation, predicting `steps_per_second`, with 100 trees.

#### Command-Line Arguments

- `--strategy`: Cross-validation strategy (default: `kfold`)
  - `split`: Simple 80-20 train-test split
  - `kfold`: K-Fold cross-validation (default)
  - `groupkfold`: GroupKFold by CPU model (prevents data leakage across hardware)

- `--target`: Target column to predict (default: `steps_per_second`)
  - Examples: `training_time_s`, `avg_ram_usage`, `max_ram_usage`, `steps_to_threshold`

- `--trees`: Number of trees in the forest (default: `100`)
  - More trees = better accuracy but slower training
  - Typical range: 50-500

- `--seed`: Random seed for reproducibility (default: `95`)
  - Use same seed to reproduce results

- `--depth`: Maximum tree depth (default: `None` = unlimited)
  - Lower values prevent overfitting
  - Typical range: 10-30

#### Examples

**Predict training time using K-Fold with 200 trees:**
```bash
python random_forest_trainer.py --strategy kfold --target training_time_s --trees 200
```

**Predict RAM usage with GroupKFold (hardware-aware splitting):**
```bash
python random_forest_trainer.py --strategy groupkfold --target avg_ram_usage --trees 150 --depth 20
```

**Quick test with simple train-test split:**
```bash
python random_forest_trainer.py --strategy split --target steps_per_second --trees 50
```

#### Output

The script will:
1. Train the model using the specified strategy
2. Display cross-validation metrics (R² score and Mean Absolute Error)
3. Show **feature importance rankings** (which hyperparameters matter most)
4. Save the trained model as `model_<target>.pkl`

### Linear Regression Analysis

Linear regression provides a simpler, interpretable baseline model for comparison.

#### Basic Usage

```bash
python linear_regressor.py
```

This predicts `training_time_s` by default.

#### Command-Line Arguments

- `--target`: Target column to predict (default: `training_time_s`)
  - Examples: `steps_per_second`, `avg_ram_usage`, `max_ram_usage`

#### Examples

**Predict steps per second:**
```bash
python linear_regressor.py --target steps_per_second
```

**Predict average RAM usage:**
```bash
python linear_regressor.py --target avg_ram_usage
```

#### Output

The script will:
1. Train a linear regression model on 80% of data
2. Display R² score (accuracy) and Mean Absolute Error
3. Save the trained model as `linear_model_<target>.pkl`

**Note:** If R² is negative, the relationship is non-linear and Random Forest is recommended.

### Choosing Between Models

| Model | Best For | Pros | Cons |
|-------|----------|------|------|
| **Random Forest** | Complex, non-linear relationships | High accuracy, feature importance analysis | Slower training, less interpretable |
| **Linear Regression** | Simple, linear relationships | Fast, interpretable coefficients | Poor for non-linear data (negative R²) |

### Prerequisites

Both scripts require the `TRAINING_DATA_PATH` environment variable set in your `.env` file:

```env
TRAINING_DATA_PATH=/path/to/your/training_data.csv
```

The data file can be in `.csv` or `.xlsx` format.

## Data Labels

### Training Run Metadata
- **RunID**: Unique identifier for each training run
- **Scene**: Name of the Unity scene being trained
- **Algorithm**: Reinforcement learning algorithm used (e.g., PPO)
- **Batch-size**: Number of samples processed per training step
- **Current-time**: Timestamp when training run starts

### Performance Metrics
- **Mean-reward**: Final mean reward from the last training batch
- **Training-time**: Total duration from start to end of training
- **Total-steps**: Cumulative number of training steps
- **Steps-per-second**: Training speed metric
- **Steps-to-threshold**: Steps required to reach mean reward of 100
- **Threshold-reached**: Boolean indicating if mean reward of 100 was achieved

### Neural Network Metrics
- **Policy-loss**: Loss value showing policy network changes during updates
- **Value-loss**: Loss value from the value function
- **Learning-rate**: Effective learning rate applied during training
- **Hyperparameter-learning-rate**: Input learning rate from hyperparameters

### System Information
- **Operating-System**: OS running the training process
- **CPU-model**: Processor model used
- **Num-cores**: Number of available CPU cores
- **Avg-system-cpu-percent**: Average CPU usage across entire system
- **Avg-tracked-cpu-percent**: Average CPU usage for training process only
- **Avg-ram-usage**: Average RAM usage during training (MB)
- **Max-ram-usage**: Peak RAM usage during training (MB)
- **Total-ram**: Total system RAM available (GB)

## Daily Workflow

### Starting Your Work Session

**macOS/Linux:**
```bash
# Navigate to project
cd ~/Documents/Uni_Projects/ml-agents

# Activate environment
conda activate mlagents  # or: source venv/bin/activate

# Run training
python run_training.py 1
```

**Windows:**
```bash
# Navigate to project
cd C:\Users\YourUsername\Documents\Uni_Projects\ml-agents

# Activate environment
conda activate mlagents  # or: venv\Scripts\activate

# Run training
python run_training.py 1
```

### Ending Your Work Session

```bash
# Deactivate environment
conda deactivate  # or: deactivate (for venv)
```

### Disable Conda Auto-activation (Optional)

If you don't want conda base environment to activate automatically:

```bash
conda config --set auto_activate_base false
```

Then manually activate `mlagents` environment when needed.

## Troubleshooting

### Apple Silicon grpcio Issues (M1/M2/M3/M4 Macs)

**Problem:** Installation fails with grpcio compilation errors or detects incompatible old grpcio version

**Solution:** Use conda-forge to install grpcio before installing ML-Agents:

```bash
# Activate your conda environment
conda activate mlagents

# Install grpcio from conda-forge
conda install grpcio=1.60.1 pytorch tensorboard -c conda-forge -c pytorch -y
conda install numpy=1.23.5 h5py protobuf=3.20.3 -c conda-forge -y

# Modify setup.py to remove old grpcio constraints (macOS/Linux)
sed -i '' 's/grpcio<=1.48.2/grpcio>=1.48.2/g' ml-agents-envs/setup.py ml-agents/setup.py

# For Windows, manually edit ml-agents-envs/setup.py and ml-agents/setup.py
# Change: "grpcio<=1.48.2" to "grpcio>=1.48.2"

# Install ML-Agents without dependencies
pip install -e ./ml-agents-envs --no-deps
pip install -e ./ml-agents --no-deps

# Install remaining dependencies manually
pip install pyyaml gym>=0.21.0 "pettingzoo==1.15.0" filelock>=3.4.0 \
  "attrs>=19.3.0" "huggingface-hub>=0.14" onnx==1.15.0 "cattrs<1.7,>=1.1.0" \
  cloudpickle Pillow six

# Install project dependencies
python -m pip install gspread oauth2client pandas python-dotenv

# Verify
mlagents-learn --help
```

### Windows grpcio Installation Issues

**Problem:** Visual C++ build tools errors when installing grpcio on Windows

**Solution:** Install pre-built grpcio from conda-forge:

```bash
# Activate your conda environment
conda activate mlagents

# Install grpcio from conda-forge
conda install grpcio -c conda-forge -y

# Then install ML-Agents
pip install -e ./ml-agents-envs
pip install -e ./ml-agents
```

### Python Version Issues

**Check current version:**
```bash
python -V
```

**Install correct version with conda:**
```bash
conda install python=3.10.12
```

**Install correct version with pyenv (macOS/Linux):**
```bash
pyenv install 3.10.12
pyenv local 3.10.12
```

### Virtual Environment Activation Issues

**macOS/Linux:** If `source venv/bin/activate` doesn't work:
```bash
. venv/bin/activate
```

**Windows PowerShell:** If you get execution policy errors:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Windows Command Prompt:** Use `.bat` file instead of `.ps1`:
```bash
venv\Scripts\activate.bat
```

### Conda Environment Not Found

**Problem:** `CondaError: Run 'conda init' before 'conda activate'`

**Solution:**

**macOS/Linux:**
```bash
# Reinitialize
~/miniconda3/bin/conda init zsh  # or bash
exec $SHELL -l
```

**Windows:**
1. Open Anaconda Prompt as Administrator
2. Run: `conda init cmd.exe` or `conda init powershell`
3. Restart your terminal

### ML-Agents Command Not Found

**macOS/Linux - Clear shell cache:**
```bash
hash -r
```

**Windows - Verify PATH:**
```bash
where python  # should show conda/venv python path
```

**Verify environment is activated:**
```bash
# macOS/Linux
which python  # should point to conda/venv python

# Windows
where python  # should point to conda/venv python

# All platforms
pip list | grep mlagents  # should show installed versions
```

### Dependency Conflicts

If you encounter pip dependency resolver warnings:
```bash
# These warnings are usually safe to ignore if mlagents-learn runs successfully
# To verify everything works:
mlagents-learn --help
python -c "import mlagents; print('OK')"
```

### Import Errors

**Problem:** `ModuleNotFoundError` when running scripts

**Solution:** Ensure you're using the correct Python interpreter:

**macOS/Linux:**
```bash
which python  # should point to your conda/venv directory
```

**Windows:**
```bash
where python  # should point to your conda/venv directory
```

**If not, deactivate and reactivate environment:**
```bash
conda deactivate
conda activate mlagents
```

### Windows PATH Issues

**Problem:** Commands not found even with environment activated

**Solution:** Add conda/venv Scripts directory to PATH:

1. Search "Environment Variables" in Windows
2. Edit "Path" under User variables
3. Add: `C:\Users\YourUsername\miniconda3\envs\mlagents\Scripts`
4. Restart terminal

## Project Structure

```

```

## Additional Resources

- [Unity ML-Agents Documentation](https://github.com/Unity-Technologies/ml-agents)
- [Unity ML-Agents Toolkit](https://unity.com/products/machine-learning-agents)
- [TensorBoard Documentation](https://www.tensorflow.org/tensorboard)
- [Conda Documentation](https://docs.conda.io)
- [Miniconda Installation](https://docs.conda.io/en/latest/miniconda.html)

## License

Copyright 2017-2021 Unity Technologies

[Apache License, Version 2.0](LICENSE.md)
