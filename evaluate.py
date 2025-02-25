import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import UCF101Dataset
from trainModel import CNNLSTM
import torchvision.transforms as transforms
import os
from sklearn.metrics import accuracy_score, classification_report

DATA_DIR = "data"
TEST_DIR = os.path.join(DATA_DIR, "test")
MODEL_PATH = "model.pth"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

classes = [d for d in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, d))]
num_classes = 45

test_dataset = UCF101Dataset(TEST_DIR, transform)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNLSTM(num_classes).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

y_true = []
y_pred = []

with torch.no_grad():
    for frames, labels in test_loader:
        frames, labels = frames.to(device), labels.to(device)
        
        outputs = model(frames)
        _, predicted = torch.max(outputs, 1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

accuracy = accuracy_score(y_true, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=classes))