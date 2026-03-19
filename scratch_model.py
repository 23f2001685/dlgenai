from torch import nn
import torch


class CRNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Dropout(p=0.2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Dropout(p=0.3),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Dropout(p=0.45)
        )


        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=64,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(in_features=128, out_features=num_classes)

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

        x, _ = self.lstm(x)

        x, _ = torch.max(x, dim=1)
        x = self.fc(x)
        return x

model_crnn = CRNN(num_classes=10)

