# Model architecture same as milestone-4
from torch import nn
import torch

class SecretNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SecretNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        
        self.gru = nn.GRU(
            input_size=2048,
            hidden_size=128,
            num_layers=8,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(in_features=256, out_features=num_classes)

    def forward(self, x):
        """
        Input:
            x: Tensor of shape (Batch, 1, 128, Time) representing Mel-spectrograms.
        Output:
            logits: Tensor of shape (Batch, num_classes) representing unnormalized class scores.
        """
        
        # TODO 4: Pass 'x' through the CNN backbone
        # Expected shape after CNN: (Batch, Channels=64, Mels=32, Time)
        x = self.cnn(x)
        
        b, c, f, t = x.shape
        
        x = x.permute(0, 3, 1, 2)
        
        x = x.reshape(b, t, -1)
        
        x, _ = self.gru(x)
        
        x, _ = torch.max(x, dim=1)
        
        x = self.fc(x)
        return x

model_oth = SecretNN(num_classes=10)