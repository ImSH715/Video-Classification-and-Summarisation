import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from models import CNNLSTM
from dataset import ImageSequenceDataset, mean, std

# dataset directory
dataset_root = "data/UCF-101-frames"
# annotation directory
annotation_file = "data/UCFTrainTestList"
sequence_length = 20
input_shape = (3, 128, 128)
# train in batch number
batch_size = 5
checkpoint_dir = './checkpoints/'
# Number of Epoch file to start testing and generate diagrams
num_epochs = 140
# Number of classes for optimise UCF-101
num_classes = 57

def get_transform():
    """
    Transform the input for resizing, tensor conversion, and normalisation by using torchvision
    """
    return transforms.Compose([
        transforms.Resize(input_shape[1:], Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def evaluate(model, loader, criterion, device):
    """
    Evaluate the model on the provided DataLoader
    Compute average loss, accuracy, and collect predictions and labels
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_labels, all_preds = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    if total == 0:
        print("No samples in DataLoader! Cannot compute loss/accuracy.")
        return 0.0, 0.0, np.array([]), np.array([])
    avg_loss = total_loss / total
    acc = correct / total * 100
    return avg_loss, acc, np.array(all_labels), np.array(all_preds)

def main():
    """
    Load test set, resotre model from checkpoint
    Evaluate performance
    Show ploted diagrams
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ImageSequenceDataset(dataset_root, annotation_file, sequence_length, transform=get_transform(), train=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    model = CNNLSTM(num_classes=num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()

    """
    Load trained model (checkpoint) file (.pth)
    """
    checkpoint = torch.load(os.path.join(checkpoint_dir, f"cnn_lstm_epoch_{num_epochs}.pth"), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    """
    Evaluate model
    Print Loss function, accuracy, precision, recall and f1-measure
    Evaluate model and print summary statistics
    """
    loss, acc, labels, preds = evaluate(model, loader, criterion, device)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {acc:.2f}%")
    print("Precision:", precision_score(labels, preds, average='weighted'))
    print("Recall:", recall_score(labels, preds, average='weighted'))
    print("f1-measure:", f1_score(labels, preds, average='weighted'))

    """
    Generate and plot confusion matrix for the predictions based on the trained model
    """
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=False, cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == '__main__':
    """
    Main function to enter the script
    """
    main()
