import os
import shutil
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import time
from .model import create_model

class CustomDataset(Dataset):
    def __init__(self, csv_file, data_folder, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.data_folder = data_folder
        self.transform = transform
        self.image_paths, self.labels = self.load_data()

    def load_data(self):
        image_paths = []
        labels = []

        for _, row in self.data_frame.iterrows():
            video_path = row['Video Path']
            watch_time = row['Watch Time']
            folder_name = os.path.splitext(os.path.basename(video_path))[0]
            video_folder = os.path.join(self.data_folder, folder_name)

            if not os.path.isdir(video_folder):
                print(f"Warning: Folder {video_folder} does not exist.")
                continue

            for file_name in os.listdir(video_folder):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(video_folder, file_name))
                    labels.append(np.clip(watch_time, 0, 1))

        return image_paths, labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float)

def run_training(csv_file, data_folder, epochs, batch_size, learning_rate, model_path):
    # GPU Metrics
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # in GB
        gpu_memory_reserved = torch.cuda.memory_reserved() / (1024 ** 3)  # in GB
        print(f"GPU Count: {gpu_count}")
        print(f"GPU Memory Allocated: {gpu_memory_allocated:.2f} GB")
        print(f"GPU Memory Reserved: {gpu_memory_reserved:.2f} GB")
        print(f"Using Device: {torch.cuda.get_device_name(device)}")
    else:
        print("CUDA is not available. Using CPU.")

    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Define datasets and dataloaders
    dataset = CustomDataset(csv_file, data_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Split dataset into training and validation sets
    num_samples = len(dataset)
    indices = list(range(num_samples))
    split = int(np.floor(0.2 * num_samples))  # 20% validation
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

    model = create_model()
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    checkpoints_dir = 'checkpoints'
    if os.path.isdir(checkpoints_dir):
        shutil.rmtree(checkpoints_dir)
    os.makedirs(checkpoints_dir, exist_ok=True)
    print(f"'{checkpoints_dir}' directory has been reset.")

    # Ensure GPU is used
    model.to(device)
    if not torch.cuda.is_available() or not next(model.parameters()).is_cuda:
        print("Warning: GPU not used")

    # Calculate the width for formatting
    epoch_width = len(str(epochs))

    loss_data = []
    val_loss_data = []
    val_accuracy_data = []

    for epoch in range(epochs):
        start_time = time.time()  # Start time of epoch
        model.train()
        running_loss = 0.0
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{epochs}', unit='batch', colour='green', ncols=100) as pbar:
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = model(images).squeeze()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # Update progress bar with loss
                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
                pbar.update()

            avg_loss = running_loss / len(train_loader)
            loss_data.append(avg_loss)
            
            # Validation phase
            model.eval()
            val_running_loss = 0.0
            correct_preds = 0
            total_preds = 0
            with torch.no_grad():
                for val_images, val_labels in val_loader:
                    val_images, val_labels = val_images.to(device), val_labels.to(device)
                    val_outputs = model(val_images).squeeze()
                    val_loss = criterion(val_outputs, val_labels)
                    val_running_loss += val_loss.item()

                    # Calculate accuracy
                    predicted = (val_outputs >= 0.5).float()
                    correct_preds += (predicted == val_labels).sum().item()
                    total_preds += len(val_labels)

            avg_val_loss = val_running_loss / len(val_loader)
            avg_val_accuracy = correct_preds / total_preds
            val_loss_data.append(avg_val_loss)
            val_accuracy_data.append(avg_val_accuracy)

            end_time = time.time()  # End time of epoch
            epoch_time = end_time - start_time  # Duration of epoch

            # Update progress bar description
            pbar.set_description(f'Epoch {epoch+1:0{epoch_width}d}/{epochs} | Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_accuracy:.4f} | Time: {epoch_time:.2f}s')

        torch.save(model.state_dict(), os.path.join(checkpoints_dir, f'model_epoch_{epoch+1:02d}.pth'))

    torch.save(model.state_dict(), model_path)

    print(f"\nTraining complete. Model saved as '{model_path}'.")

    return loss_data, val_loss_data, val_accuracy_data
