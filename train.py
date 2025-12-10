import os
import time
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from models import CNNLSTM
from dataset import ImageSequenceDataset, mean, std

# dataset directory
dataset_root = "data/UCF-101-frames" 
# annotation file (train and test set in list)
annotation_file = "data/UCFTrainTestList" 

sequence_length = 20
input_shape = (3, 128, 128)
# Reduce batch size to reduce train runtime
batch_size = 5 
# Checkpoint to start from, ex) cnn_lstm_epoch_144.pth
checkpoint = 140  
# Ending epoch number
num_epochs = 200
learning_rate = 1e-4
# directory to save checkpoint model
checkpoint_dir = './checkpoints/' 
# number of classes for optimised UCF-101
num_classes = 57


os.makedirs(checkpoint_dir, exist_ok=True)

def get_transform():
    """
    Transform the input for resizing, tensor conversion, and normalisation by using torchvision
    """
    return transforms.Compose([
        transforms.Resize(input_shape[1:], Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

def main():
    """
    Main train function
    """

    """
    Load train dataset and dataloaders
    """
    train_dataset = ImageSequenceDataset(dataset_root, annotation_file, sequence_length, transform=get_transform(), train=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNLSTM(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    """
    Training continue from the configured checkpoint file (epoch number)
    """
    start_epoch = 0
    if checkpoint > 0:
        checkpoint_path = os.path.join(checkpoint_dir, f"cnn_lstm_epoch_{checkpoint}.pth")
        if os.path.exists(checkpoint_path):
            checkpoint_data = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint_data['model_state_dict'])
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            start_epoch = checkpoint
            print(f"\nResuming from checkpoint: {checkpoint_path}")
            print(f"Resuming at epoch {start_epoch + 1}/{num_epochs}\n")
        else:
            print(f"Checkpoint file not found.")
    else:
        print("Starting training from 0")

    """
    Loop over epochs for the training
    """
    for epoch in range(start_epoch + 1, num_epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        epoch_start_time = time.time()
        num_batches = len(train_loader)

        print(f"\nEpoch [{epoch}/{num_epochs}] started.")

        for batch_idx, (images, labels) in enumerate(train_loader):
            batch_start_time = time.time()
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            """
            Prining Batch number, Loss fuction, Accuracy, and ETA for each batch
            """
            batch_loss = loss.item()
            batch_acc = (preds == labels).float().mean().item() * 100
            elapsed = time.time() - batch_start_time
            batches_left = num_batches - (batch_idx + 1)
            eta = elapsed * batches_left

            print(f"Batch [{batch_idx+1}/{num_batches}] "
                  f"Loss: {batch_loss:.4f} "
                  f"Acc: {batch_acc:.2f}% "
                  f"ETA: {int(eta//60)}m {int(eta%60)}s")
        """
        Printing Loss function, Accuracy, and ETA for each epoch
        """
        epoch_loss = running_loss / total
        epoch_acc = correct / total * 100
        epoch_time = time.time() - epoch_start_time
        print(f"\nEpoch [{epoch}/{num_epochs}] completed. "
              f"Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}% "
              f"Time: {int(epoch_time//60)}m {int(epoch_time%60)}s")

        """
        Save trained model into (.pth) file, after each epoch in the name of cnn_lstm_epoch(epoch_number).pth
        """
        checkpoint_save_path = os.path.join(checkpoint_dir, f"cnn_lstm_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_save_path)

if __name__ == '__main__':
    # main function to start the script
    main()
