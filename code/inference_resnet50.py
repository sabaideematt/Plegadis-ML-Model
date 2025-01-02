# inference.py
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError
import io
import os
import numpy as np
import logging
import traceback

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create handler for logging to stdout
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
handler.setFormatter(formatter)
logger.addHandler(handler)

def model_fn(model_dir):
    """
    Load the PyTorch model from the `model_dir`.

    Args:
        model_dir (str): Directory where the model artifacts are stored.

    Returns:
        torch.nn.Module: The loaded PyTorch model.
    """
    logger.info(f"INFERENCE LOG :: Loading model from {model_dir}/model.pth")

    # Determine the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"INFERENCE LOG :: Using device: {device}")

    # Instantiate ResNet-50 architecture
    try:
        model = models.resnet50(pretrained=False)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 2)  # Binary classification: birds / no birds
        logger.info("INFERENCE LOG :: ResNet-50 model instantiated.")
    except Exception as e:
        logger.error(f"INFERENCE LOG :: Error instantiating ResNet-50 model: {e}")
        logger.error(traceback.format_exc())
        raise e

    # Load the state dictionary
    model_path = os.path.join(model_dir, 'model.pth')
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        logger.info("INFERENCE LOG :: Model state dictionary loaded successfully.")
    except Exception as e:
        logger.error(f"INFERENCE LOG :: Error loading the model state dictionary: {e}")
        logger.error(traceback.format_exc())
        raise e

    # Move the model to the appropriate device and set to evaluation mode
    model.to(device)
    model.eval()
    logger.info("INFERENCE LOG :: Model loaded and set to evaluation mode.")

    # Log model architecture (optional)
    logger.info("INFERENCE LOG :: Model architecture:")
    for name, param in model.named_parameters():
        logger.info(f"INFERENCE LOG :: Layer {name}, Params: {param.shape}")

    return model

def input_fn(request_body, content_type):
    """
    Deserialize and preprocess the input data.

    Args:
        request_body (bytes): The input data.
        content_type (str): The MIME type of the input data.

    Returns:
        torch.Tensor: The preprocessed input tensor.
    """
    logger.info(f"INFERENCE LOG :: Received content type: {content_type}")
    logger.info(f"INFERENCE LOG :: First 10 bytes of request body: {request_body[:10]}")

    try:
        if content_type == 'image/jpeg':
            logger.info("INFERENCE LOG :: Processing image/jpeg content type.")
            # Load the image
            img = Image.open(io.BytesIO(request_body)).convert('RGB')
            logger.info("INFERENCE LOG :: Image loaded successfully.")

            # Define preprocessing transformations
            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])

            # Apply preprocessing
            img_t = preprocess(img)
            img_t = img_t.unsqueeze(0)  # Add batch dimension
            logger.info(f"INFERENCE LOG :: Processed image tensor shape: {img_t.shape}")
            return img_t

        elif content_type == 'application/x-npy':
            logger.info("INFERENCE LOG :: Processing application/x-npy content type.")
            # Deserialize the NumPy array
            np_array = np.load(io.BytesIO(request_body), allow_pickle=True)
            tensor = torch.tensor(np_array, dtype=torch.float32)
            logger.info(f"INFERENCE LOG :: Received NumPy array shape: {tensor.shape}")
            tensor = tensor.unsqueeze(0)  # Add batch dimension if needed
            return tensor

        elif content_type == 'application/x-torch':
            logger.info("INFERENCE LOG :: Processing application/x-torch content type.")
            # Deserialize the preprocessed tensor
            tensor = torch.load(io.BytesIO(request_body), map_location='cpu')
            logger.info(f"INFERENCE LOG :: Received tensor shape: {tensor.shape}")
            return tensor

        else:
            error_msg = f"INFERENCE LOG :: Unsupported content type: {content_type}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    except UnidentifiedImageError as e:
        logger.error(f"INFERENCE LOG :: Error identifying image: {e}")
        logger.error(traceback.format_exc())
        raise ValueError(f"INFERENCE LOG :: Error identifying image: {e}")
    except Exception as e:
        logger.error(f"INFERENCE LOG :: Error during input processing: {e}")
        logger.error(traceback.format_exc())
        raise ValueError(f"INFERENCE LOG :: Error during input processing: {e}")

def predict_fn(input_data, model):
    """
    Perform inference on the input data using the loaded model.

    Args:
        input_data (torch.Tensor): The preprocessed input tensor.
        model (torch.nn.Module): The loaded PyTorch model.

    Returns:
        dict: A dictionary containing the predicted class and probabilities.
    """
    logger.info("INFERENCE LOG :: Starting prediction.")

    # Determine the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"INFERENCE LOG :: Using device for inference: {device}")

    # Move input data to the appropriate device and model to the device
    input_data = input_data.to(device)
    model = model.to(device)

    with torch.no_grad():
        try:
            outputs = model(input_data)
            logger.info(f"INFERENCE LOG :: Raw model outputs: {outputs}")

            probabilities = torch.softmax(outputs, dim=1)
            _, predicted_class = torch.max(probabilities, 1)

            logger.info(f"INFERENCE LOG :: Probabilities: {probabilities}")
            logger.info(f"INFERENCE LOG :: Predicted class: {predicted_class.item()}")

            return {
                'predicted_class': predicted_class.item(),
                'probabilities': probabilities.cpu().numpy().tolist()
            }

        except Exception as e:
            logger.error(f"INFERENCE LOG :: Error during model prediction: {e}")
            logger.error(traceback.format_exc())
            raise e

def output_fn(prediction, accept):
    """
    Serialize the prediction output.

    Args:
        prediction (dict): The prediction output from predict_fn.
        accept (str): The MIME type for the response.

    Returns:
        bytes: The serialized prediction output.
    """
    logger.info(f"INFERENCE LOG :: Received accept type: {accept}")

    if accept == "application/json":
        logger.info("INFERENCE LOG :: Serializing prediction to application/json")
        return json.dumps(prediction).encode('utf-8')
    elif accept == "application/text":
        logger.info("INFERENCE LOG :: Serializing prediction to application/text")
        return str(prediction).encode('utf-8')
    else:
        error_msg = f"INFERENCE LOG :: Unsupported accept type: {accept}"
        logger.error(error_msg)
        raise ValueError(error_msg)

import json  # Added import for json serialization

print("INFERENCE LOG :: Inference script loaded successfully")
