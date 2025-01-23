# Plegadis-ML-Model

## Overview

This project was developed for the **Case Studies in Machine Learning** graduate course at **UT Austin, Fall 2024**, focusing on classifying images of Plegadis birds using machine learning models deployed via AWS SageMaker. The dataset used for this project is courtesy of **Ryan Klutts of Louisiana State University**. The goal of the project was to compare a custom Convolutional Neural Network (CNN) model with transfer learning models using **EfficientNet** and **ResNet-50** architectures. Ultimately, I decided to use the ResNet-50 based model for classification since it performed the best on the validation set. The models are implemented in the files:

- `code/train.py`, `code/model.py`, and `code/inference.py` – Custom CNN model implementation
- `code/train_efficient_net.py` and `code/inference_efficient_net.py` – Transfer learning with EfficientNet
- `code/train_with_resnet50.py` and `code/inference_resnet50.py` – Transfer learning with ResNet-50

## Project Structure

- **code/** – Source code for preprocessing and training
- **sagemaker\_env/** – AWS environment setup
- **Notebooks:**
  - `preprocessor.ipynb` – Data preparation
  - `training_job.ipynb` – Model training
  - `PyTorchModel.ipynb` – Deploy the model to an inference endpoint, call, and analyze

## Requirements

- Python 3.x
- PyTorch
- AWS SageMaker SDK

## Usage

1. Clone the repository:
   ```
   git clone https://github.com/sabaideematt/Plegadis-ML-Model.git
   ```
2. Set up AWS credentials and run the training notebook.


## Note
There is a fair amount of junk in this repo. I basically just saved what was in AWS and uploaded it to GitHub raw. 


