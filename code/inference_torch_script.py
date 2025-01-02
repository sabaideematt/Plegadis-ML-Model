# inference.py
import torch
from torchvision import transforms
from PIL import Image
import io
import numpy as np

# Function to load the TorchScript model
def model_fn(model_dir):
    print(f"INFERENCE LOG :: Loading TorchScript model from {model_dir}/model.pt")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.jit.load(f'{model_dir}/model.pt', map_location=device)  # Load the TorchScript model
    model.to(device)
    model.eval()
    print("INFERENCE LOG :: TorchScript Model loaded successfully")
    return model

# Function to preprocess the input image or tensor
def input_fn(request_body, content_type):
    try:
        print(f"INFERENCE LOG :: Received content type: {content_type}")
        
        if content_type == 'image/jpeg':
            print("INFERENCE LOG :: Processing image file...")
            # Load and preprocess the image
            img = Image.open(io.BytesIO(request_body)).convert('RGB')
            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], 
                                     [0.229, 0.224, 0.225])
            ])
            img_t = preprocess(img)
            img_t = img_t.unsqueeze(0)  # Add batch dimension
            print(f"INFERENCE LOG :: Processed image shape: {img_t.shape}")
            return img_t

        elif content_type == 'application/x-npy':
            print("INFERENCE LOG :: Receiving NumPy array...")
            stream = io.BytesIO(request_body)
            np_array = np.load(stream, allow_pickle=True)
            tensor = torch.tensor(np_array, dtype=torch.float32)
            tensor = tensor.unsqueeze(0)  # Add batch dimension if needed
            print(f"INFERENCE LOG :: Received NumPy array shape: {tensor.shape}")
            return tensor

        elif content_type == 'application/x-torch':
            print("INFERENCE LOG :: Receiving preprocessed tensor...")
            tensor = torch.load(io.BytesIO(request_body), map_location='cpu')
            print(f"INFERENCE LOG :: Received tensor shape: {tensor.shape}")
            return tensor

        else:
            raise ValueError(f"INFERENCE LOG :: Unsupported content type: {content_type}")
    
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        raise ValueError(f"INFERENCE LOG :: Error during input processing: {e}\n{traceback_str}")

# Function to perform inference
def predict_fn(input_data, model):
    print("INFERENCE LOG :: Performing inference...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_data = input_data.to(device)
    model = model.to(device)
    with torch.no_grad():
        output = model(input_data)
        probabilities = torch.softmax(output, dim=1)
        _, predicted_class = torch.max(probabilities, 1)
        print(f"INFERENCE LOG :: Inference output shape: {output.shape}")
        print(f"INFERENCE LOG :: Probabilities shape: {probabilities.shape}")
    return {'predicted_class': predicted_class.item(), 'probabilities': probabilities.cpu().numpy().tolist()}

print("INFERENCE LOG :: Inference script loaded successfully")
