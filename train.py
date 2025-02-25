import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import UCF101Dataset
from trainModel import CNNLSTM
import torchvision.transforms as transforms
import os

DATA_DIR = "data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

batch_size = 8
num_classes = len([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))])
num_epochs = 10
learning_rate = 0.001

# Image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialize the dataset and dataloaders
train_dataset = UCF101Dataset(TRAIN_DIR, transform)
test_dataset = UCF101Dataset(TEST_DIR, transform)

# Ensure datasets have loaded properly and avoid reloading
if len(train_dataset) == 0:
    print("❌ ERROR: No data found in training directory.")
    exit()
if len(test_dataset) == 0:
    print("❌ ERROR: No data found in testing directory.")
    exit()

# DataLoader initialization (only once)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNLSTM(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Initialize best_loss to a very high value
best_loss = float('inf')  # or a large number

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    # Iterating over batches in the dataset (should happen only once per epoch)
    for frames, labels in train_loader:
        frames, labels = frames.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(frames)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # Track best loss (for checkpointing or early stopping)
    if avg_loss < best_loss:
        best_loss = avg_loss
        print(f"✨ New best loss: {best_loss:.4f} (Saving model)")

        # Save the model
        torch.save(model.state_dict(), "best_model.pth")

# Save model after training
torch.save(model.state_dict(), "model.pth")
