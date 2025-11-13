
# Few-Shot Machine Unlearning Using Our OES Framework

## Overview

This project includes several Python scripts designed to perform machine unlearning using our proposed GFOES framework. The project utilizes PyTorch for deep learning tasks and includes custom implementations for specific models, training procedures, and data handling.

---

## File Descriptions

### `parameter.py`
- **Purpose**: Contains all the global parameters and configurations used across the project, such as dataset paths, model hyperparameters, and other constants.
- **Key Features**:
  - Centralized configuration management.
  - Defines paths, model parameters, and dataset-related constants.
  - Provides easy customization of training settings and hyperparameters.

### `data_utils.py`
- **Purpose**: Contains utility functions for handling datasets, including data loading, preprocessing, and filtering based on specific conditions or criteria.
- **Key Features**:
  - Functions to select samples from specific classes in a dataset.
  - Preprocessing functions like data normalization and transformation.

### `function.py`
- **Purpose**: Provides essential functions to support model training and evaluation.
- **Key Features**:
  - **Model Evaluation**: The `evaluate` function calculates average loss and accuracy on a validation set.
  - **Training Functionality**: The `train_model` function manages the training process, including gradient clipping and periodic evaluation.

 
### `model_utils.py`
- **Purpose**: Contains definitions for various neural network models used in the project, including the custom `AllCNN` and `ResNet` models.
- **Key Features**:
  - Implementation of custom neural networks.
  - Support for both AllCNN and ResNet18 and ResNet50 architectures.


### `original_model.py`
- **Purpose**: Script for training the original model before applying our OES framework or retraining methods. The trained model serves as the baseline for comparison.
- **Key Features**:
  - Original training loop and model setup.
  - Saves the model state for later use in unlearning or retraining processes.

### `method_creatrain.py`
- **Purpose**: Implements a baseline method called "Retrain". 
- **Key Features**:
  - Baseline model training functions.
  - Integration with standard data loaders and optimizers.
 
### `GFN.py`
- **Purpose**: Script for training a specific generative model called "GAFN". This script handles the training loop, model setup, and the overall training process.
- **Key Features**:
  - Training loop for GAFN.
  - Model checkpointing and performance evaluation during training.

### `OES_unlearning.py`
- **Purpose**: Implements the "OES Unlearning" framework, which handles the unlearning process where specific information is removed from a trained model.
- **Key Features**:
  - Functions for performing unlearning on the model.
  - Integration with GAFN and other methods.

---


## Experiment Environment

The experiments are executed on a server equipped with an NVIDIA GeForce RTX 3090 GPU, 128GB of RAM, and Ubuntu 20.10 as the operating system.The Python version used is Python 3.10.
In our experiments, the indices of the classes to be forgotten in the four unlearning scenarios are as follows: (3rd), (3rd, 4th), (3rd, 4th, 5th, 6th), (1st, 2nd, 3rd, 4th, 5th, 6th, 7th).

---
## Environment Setup

To recreate the environment used for this project, you can install the dependencies listed in `requirements.txt`.


---

## How to Use

### 1. Setup
- Ensure all dependencies, particularly PyTorch and torchvision, are installed.
- Adjust paths and configurations in `parameter.py` as needed to match your environment.

### 2. Training the Original Model
- Run `original_model.py` to train the initial model. This model will be used as a baseline before applying any unlearning or retraining methods.

### 3. CRetrain Baseline
- Run `method_reatrain.py` to train the model using the RETRAIN baseline method.

### 4. GAFN Training
- Run `GFN.py` to train the Generator.

### 5. Performing Unlearning
- Run `OES_unlearning.py` to apply the unlearning framework to the trained model. This step removes specific data-related knowledge from the model.



---

## Customization

- **Models**: You can add or modify model architectures in `model_utils.py`.
- **Parameters**: All adjustable parameters, such as learning rates, batch sizes, and dataset paths, are defined in `parameter.py`.
