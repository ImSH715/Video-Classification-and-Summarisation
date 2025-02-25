import torch
import torch.nn as nn
import torchvision.models as models

class CNNLSTM(nn.Module):
    def __init__(self, num_classes):
        super(CNNLSTM, self).__init__()
        self.cnn = models.resnet50(pretrained=True)
        self.cnn.fc = nn.Identity()  # Remove FC layer

        self.lstm = nn.LSTM(input_size=2048, hidden_size=512, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(512 * 2, num_classes)

    def forward(self, x):
        batch_size, seq_length, C, H, W = x.shape
        x = x.view(batch_size * seq_length, C, H, W)

        x = self.cnn(x)
        x = x.view(batch_size, seq_length, -1)

        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])  

        return x
