import torch
import torch.nn as nn
import torch.nn.functional as F

class ReNN(nn.Module):
    def __init__(self, input_size):
        super(ReNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)