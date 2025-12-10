import torch.nn as nn
from torchvision import models

class CNNLSTM(nn.Module):
    def __init__(
        # total number of classes
        self, num_classes,
        # number of features in the hidden state
        hidden_dim=256,
        # number of LSTM layers
        lstm_layers=1, 
        # disable back and forward LSTM process
        bidirectional=False
    ):
        """
        Initialise the CNN-LSTM model,
        pretrained ResNet-18 as CNN feature extractor
        LSTM to process temporal features
        Final Linear layer for classification
        """
        super(CNNLSTM, self).__init__()
        
        """
        CNN feature extractor using ResNet18 for feature extraction
        Only the feature layers are used
        """
        resnet = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1]) 
        # Output features from ResNet
        self.conv_out_dim = resnet.fc.in_features  
        
        # LSTM module for temporal modeling across frame features
        self.lstm = nn.LSTM(
            self.conv_out_dim, hidden_dim, lstm_layers,
            batch_first=True, bidirectional=bidirectional
        )
        
        """
        Output LSTM layer mapping output to number of action classes
        """
        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.output_layers = nn.Linear(lstm_out_dim, num_classes)

    def forward(self, x):
        """
        Pass the reshpaed input and pass each frame through CNN,
        collect sequence of features and process with LSTM,
        use last LSTM output for classification
        """
        b, t, c, h, w = x.shape
        
        """
        Transform input to fed all frames from all sequences into the CNN together.
        """
        x = x.view(b * t, c, h, w)
        """
        Pass to cNN feature extractor
        """
        x = self.feature_extractor(x)
        x = x.view(b, t, -1)
        
        """
        Pass the output to the LSTM
        """
        x, _ = self.lstm(x)
        
        """
        LSTM output is used at the last time step for classification
        """
        x = x[:, -1, :]
        
        """
        Final classification layer returns logits for all classes
        """
        return self.output_layers(x)
