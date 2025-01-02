# train_efficient_net.py
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

import os
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score

# Import EfficientNet
from torchvision.models import efficientnet_b0

def train():
    # Define the S3 bucket and prefix (for train and validation data)
    TRAIN_S3_BUCKET = 'csml-data-bucket'
    TRAIN_S3_PREFIX = 'preprocessed/train'
    VALIDATION_S3_PREFIX = 'preprocessed/validation'

    # Step 1: Verify S3 bucket access (logging step)
    s3 = boto3.resource('s3')
    try:
        print(f" TRAIN LOG :: Attempting to access S3 bucket: {TRAIN_S3_BUCKET}...")
        bucket = s3.Bucket(TRAIN_S3_BUCKET)
        objects = list(bucket.objects.filter(Prefix=TRAIN_S3_PREFIX))
        if not objects:
            raise FileNotFoundError(f"No objects found in s3://{TRAIN_S3_BUCKET}/{TRAIN_S3_PREFIX}")
        print(f" TRAIN LOG :: Successfully accessed S3 bucket: {TRAIN_S3_BUCKET} and found {len(objects)} objects in the training folder.")
    except (NoCredentialsError, ClientError) as e:
        raise Exception(f" TRAIN LOG :: Could not access S3 bucket: {e}")

    # Step 2: Setup training and validation data transforms
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    validation_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    train_dir = '/opt/ml/input/data/training'
    validation_dir = '/opt/ml/input/data/validation'

    # Verify directories exist
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f" TRAIN LOG :: Training directory {train_dir} does not exist.")

    if not os.path.exists(validation_dir):
        raise FileNotFoundError(f" TRAIN LOG :: Validation directory {validation_dir} does not exist.")

    # Load datasets with respective transforms
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    validation_dataset = datasets.ImageFolder(root=validation_dir, transform=validation_transform)

    # Compute class weights for imbalanced dataset handling
    class_counts = Counter(train_dataset.targets)
    total_samples = sum(class_counts.values())
    class_weights = [total_samples / class_counts[i] for i in range(len(class_counts))]
    class_weights = torch.FloatTensor(class_weights).to('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataloaders with stratified sampling for training
    targets = train_dataset.targets
    weights_samples = [class_weights[i] for i in targets]
    sampler = torch.utils.data.WeightedRandomSampler(weights_samples, num_samples=len(weights_samples), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler, num_workers=4)
    validation_loader = DataLoader(validation_dataset, batch_size=16, shuffle=False, num_workers=4)

    # Instantiate the EfficientNet model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = len(train_dataset.classes)

    # Load the pre-trained EfficientNet model
    model = efficientnet_b0(weights='IMAGENET1K_V1').to(device)

    # Replace the classifier with a new one for our number of classes
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes).to(device)

    # Define loss criterion with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Define optimizer and learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Training loop parameters
    best_val_loss = float('inf')
    patience = 3
    trigger_times = 0
    epochs = 20  # Increased number of epochs

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        scheduler.step()

        # Validation loop
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for images, labels in validation_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

        avg_val_loss = val_loss / len(validation_loader)
        val_accuracy = 100 * correct / total

        # Compute additional metrics
        average = 'binary' if num_classes == 2 else 'macro'
        precision = precision_score(all_labels, all_preds, average=average, zero_division=0)
        recall = recall_score(all_labels, all_preds, average=average, zero_division=0)
        f1 = f1_score(all_labels, all_preds, average=average, zero_division=0)

        print(f" EPOCH {epoch + 1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        print(f" VALIDATION METRICS :: Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}")

        # Early stopping mechanism
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            trigger_times = 0
            # Save the best model
            torch.save(model.state_dict(), 'best_model.pth')
            print(f" TRAIN LOG :: Best model saved at epoch {epoch + 1}")
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(' TRAIN LOG :: Early stopping triggered!')
                break

    # Save the final model
    save_model(model)

def save_model(model):
    # SM_MODEL_DIR is the directory where SageMaker saves the model
    model_dir = os.environ.get('SM_MODEL_DIR', '.')
    model_path = os.path.join(model_dir, 'model.pth')

    # Ensure the model directory exists
    os.makedirs(model_dir, exist_ok=True)

    # Save the model
    torch.save(model.state_dict(), model_path)
    print(f" TRAIN LOG :: Model saved to {model_path}")

if __name__ == "__main__":
    train()
