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
    self.fc1 = nn.Linear(28 * 28, 50 * 50)
    self.fc2 = nn.Linear(50 * 50, 25 * 25)
    self.fc3 = nn.Linear(25 * 25, 13 * 13)
    self.fc4 = nn.Linear(13 * 13, 6 * 6)
    self.output = nn.Linear(6 * 6, 10)
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

class LSTMnet(nn.Module):
  def __init__(self):
    super().__init__()
    self.lstm = nn.LSTM(input_size = 28, hidden_size = 128, batch_first = True, num_layers = 2)
    self.output = nn.Linear(128, 10)
  def forward(self, x):
    x = x.squeeze(1)
    x, _ = self.lstm(x)
    x = x[:, -1, :]
    x = self.output(x)
    return x

class ViTpreprocess(nn.Module):
  def __init__(self, embed_dim = 64):
    super().__init__()
    self.cnn_preprocess = nn.Conv2d(in_channels = 1, out_channels = embed_dim, kernel_size = 4, stride = 4)
    self.flatten = nn.Flatten(start_dim = 2)
    self.cls_token = nn.Parameter(torch.randn(1 , 1, embed_dim) * 0.02)
    self.pos_encode = nn.Parameter(torch.randn(1, 7 * 7 + 1, embed_dim) * 0.02)

  def forward(self, x):
    batch_size = x.shape[0]
    patches = self.cnn_preprocess(x)  
    flt_patches = self.flatten(patches) 
    perm_patches = torch.permute(flt_patches, (0, 2, 1)) 
    exp_cls = self.cls_token.expand(batch_size, -1, -1) 
    pre_vit = torch.cat((exp_cls, perm_patches), dim = 1) 
    vit_input = pre_vit + self.pos_encode
    return vit_input

class VisionTransformer(nn.Module):
  def __init__(self, embed_dim = 64, num_layers = 6, num_heads = 4, num_classes = 10):
    super().__init__()
    self.vit_preprocess = ViTpreprocess(embed_dim = embed_dim)
    encoder_layer = nn.TransformerEncoderLayer(d_model = embed_dim, nhead = num_heads, batch_first = True )
    self.encoder = nn.TransformerEncoder(encoder_layer = encoder_layer, num_layers = num_layers)
    self.classifier = nn.Linear(in_features = embed_dim, out_features = num_classes)
  
  def forward(self, x):
    x = self.vit_preprocess(x)
    transformer_output = self.encoder(x)
    cls_token_output = transformer_output[:, 0, :]
    logits = self.classifier(cls_token_output) 
    return logits
    


    



