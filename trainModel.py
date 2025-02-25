import torch
import torch.nn as nn
import torchvision.models as models

class CNNLSTM(nn.Module):
    def __init__(self, num_classes):
        super(CNNLSTM, self).__init__()

        # Use ResNet18 instead of ResNet50
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Identity()  # Remove final FC layer

        # Reduce LSTM complexity: 1 layer, hidden_size=256
        self.lstm = nn.LSTM(input_size=512, hidden_size=256, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(256 * 2, num_classes)

    def forward(self, x):
        batch_size, seq_length, C, H, W = x.shape
        x = x.view(batch_size * seq_length, C, H, W)

        x = self.cnn(x)
        x = x.view(batch_size, seq_length, -1)

        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])  # We only care about the last output of the LSTM

        return x
