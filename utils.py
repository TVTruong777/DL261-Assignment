from google.colab import drive
import torch.nn as nn
import torch.nn.functional as F

drive.mount('/content/drive')

class LinearModel(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    self.flatten = nn.Flatten()
    self.output = nn.Linear(1 * 28 * 28, 10)
  def forward(self, x):
    x = self.flatten(x)
    x = self.output(x)
    return x

class MLP(nn.Module):
  def __init__(self):
    super().__init__()
    self.flatten = nn.Flatten()
    self.fc1 = nn.Linear(28 * 28, 45)
    self.fc2 = nn.Linear(45, 15)
    self.output = nn.Linear(15, 10)
  def forward(self, x):
    x = self.flatten(x)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.output(x)
    return x
