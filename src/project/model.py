import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # conv layers
        self.conv1 = nn.Conv2d(1, 32, 3, 1)   # (N,1,28,28) -> (N,32,26,26)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)  # -> (N,64,24,24)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        # после двух сверток и MaxPool2d(2) размер: 24x24 -> pool -> 12x12
        # поэтому 64 * 12 * 12 = 9216
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)
