from google.colab import drive
import torch.nn as nn
import torch.nn.functional as F
import torch

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
    self.fc1 = nn.Linear(28 * 28, 100 * 100)
    self.fc2 = nn.Linear(100 * 100, 50 * 50)
    self.fc3 = nn.Linear(50 * 50, 25 * 25)
    self.fc4 = nn.Linear(25 * 25, 10 * 10)
    self.output = nn.Linear(10 * 10, 10)
  def forward(self, x):
    x = self.flatten(x)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    x = F.relu(self.fc4(x))
    x = self.output(x)
    return x

class ConvNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
    self.conv_bn = nn.BatchNorm2d(num_features = 32)
    self.fc1 = nn.Linear(14 * 14 * 32, 7 * 7 * 16)
    self.bn1 = nn.BatchNorm1d(num_features = 7 * 7 * 16) 
    self.output = nn.Linear(7 * 7 * 16, 10)
  def forward(self, x):
    x = self.conv_bn(self.conv(x))
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size = 2, stride = 2) 
    x = torch.flatten(x, start_dim = 1)
    x = self.bn1(self.fc1(x))
    x = F.relu(x)
    x = self.output(x)
    return x 
