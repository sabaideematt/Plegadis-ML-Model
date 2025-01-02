# inference.py
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import numpy as np

# Import EfficientNet
from torchvision.models import efficientnet_b0

# Function to load the model
def model_fn(model_dir):
    print(f" INFERENCE LOG :: Loading model from {model_dir}/model.pth")
    # Determine the device to load the model on
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the pre-trained EfficientNet model without pre-trained weights
    model = efficientnet_b0(weights=None)

    # Load the state dict
    state_dict = torch.load(f'{model_dir}/model.pth', map_location=device)

    # Modify the classifier to match the number of classes
    # Infer num_classes from the classifier layer's weight tensor
    num_classes = state_dict['classifier.1.weight'].shape[0]
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    # Load the state dict into the model
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print("INFERENCE LOG :: Model loaded, printing model parameters:")
    for name, param in model.named_parameters():
        print(f"INFERENCE LOG :: Layer {name}, Params: {param.shape}")
    print("INFERENCE LOG :: Model loaded successfully")
    return model

# Function to preprocess the input image or tensor
def input_fn(request_body, content_type):
    try:
        print(f" INFERENCE LOG :: Received content type: {content_type}")
        # Log the initial bytes (converted to string for logging purposes)
        print(f" INFERENCE LOG :: First 10 bytes of request body: {str(request_body[:10])}")  
        
        if content_type == 'image/jpeg':
            print(" INFERENCE LOG :: Receiving image file...")
            # Load and preprocess the image
            img = Image.open(io.BytesIO(request_body)).convert('RGB')

            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
            img_t = preprocess(img)
            img_t = img_t.unsqueeze(0)  # Add batch dimension
            print(f" INFERENCE LOG :: Processed image shape: {img_t.shape}")
            return img_t

        elif content_type == 'application/x-npy':
            print(" INFERENCE LOG :: Receiving NumPy array...")
            # Deserialize the NumPy array
            stream = io.BytesIO(request_body)
            np_array = np.load(stream, allow_pickle=True)
            tensor = torch.tensor(np_array, dtype=torch.float32)
            print(f" INFERENCE LOG :: Received NumPy array shape: {tensor.shape}")
            tensor = tensor.unsqueeze(0)  # Add batch dimension if needed
            return tensor

        elif content_type == 'application/x-torch':
            print(" INFERENCE LOG :: Receiving preprocessed tensor...")
            # Deserialize the preprocessed tensor
            tensor = torch.load(io.BytesIO(request_body), map_location='cpu')
            print(f" INFERENCE LOG :: Received tensor shape: {tensor.shape}")
            return tensor

        else:
            raise ValueError(f' INFERENCE LOG :: Unsupported content type: {content_type}')
    
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        raise ValueError(f" INFERENCE LOG :: Error during input processing: {e}\n{traceback_str}")

# Function to perform inference
def predict_fn(input_data, model):
    print(" INFERENCE LOG :: Performing inference...")
    # Determine the device to perform inference on
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_data = input_data.to(device)
    model.to(device)
    with torch.no_grad():
        output = model(input_data)
        probabilities = torch.softmax(output, dim=1)
        _, predicted_class = torch.max(probabilities, 1)
        print(f" INFERENCE LOG :: Inference output: {output}")
        print(f" INFERENCE LOG :: Probabilities: {probabilities}")
    return {'predicted_class': predicted_class.item(), 'probabilities': probabilities.cpu().numpy().tolist()}

print("INFERENCE LOG :: Inference script loaded successfully")
