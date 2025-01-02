# train.py
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

import os
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from collections import Counter
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score
import tarfile
import shutil
from tempfile import TemporaryDirectory

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
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ])

    validation_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
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
    class_weights = torch.FloatTensor(class_weights)

    # Dataloaders with stratified sampling for training
    targets = train_dataset.targets
    weights = [class_weights[i] for i in targets]
    sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler, num_workers=4)
    validation_loader = DataLoader(validation_dataset, batch_size=16, shuffle=False, num_workers=4)

    # Instantiate the ResNet-50 model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.resnet50(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)  # Assuming binary classification: bird / no bird
    model = model.to(device)

    # Define loss criterion with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # Define optimizer and learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
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
        precision = precision_score(all_labels, all_preds, average='binary')
        recall = recall_score(all_labels, all_preds, average='binary')
        f1 = f1_score(all_labels, all_preds, average='binary')

        print(f" EPOCH {epoch + 1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        print(f" VALIDATION METRICS :: Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}")

        # Early stopping mechanism
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            trigger_times = 0
            # Save the best model
            save_model(model, best=True)
            print(f" TRAIN LOG :: Best model saved at epoch {epoch + 1}")
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(' TRAIN LOG :: Early stopping triggered!')
                break

    # Save the final model
    save_model(model, best=False)

def save_model(model, best=False):
    # SM_MODEL_DIR is the directory where SageMaker saves the model
    model_dir = os.environ.get('SM_MODEL_DIR', '.')
    
    # Save the TorchScript model
    model_scripted_path = os.path.join(model_dir, 'model_scripted.pt')
    model.eval()  # Set the model to evaluation mode
    scripted_model = torch.jit.script(model)  # Script the model
    scripted_model.save(model_scripted_path)
    print(f" TRAIN LOG :: TorchScript model saved to {model_scripted_path}")
    
    # Package model into model.tar.gz for SageMaker
    with TemporaryDirectory() as temp_dir:
        # Copy TorchScript model into temporary directory
        shutil.copy(model_scripted_path, os.path.join(temp_dir, 'model_scripted.pt'))
        
        # Create tar.gz file
        model_tar_path = os.path.join(model_dir, 'model.tar.gz')
        with tarfile.open(model_tar_path, 'w:gz') as tar:
            tar.add(os.path.join(temp_dir, 'model_scripted.pt'), arcname='model_scripted.pt')
            
        if best:
            print(f" TRAIN LOG :: Best model packaged as {model_tar_path}")
        else:
            print(f" TRAIN LOG :: Final model packaged as {model_tar_path}")

if __name__ == "__main__":
    train()
