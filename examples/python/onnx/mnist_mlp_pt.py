import torch
import torch.nn as nn

class MLP(nn.Module):
  def __init__(self):
    super().__init__()
    self.linear1 = nn.Linear(784, 512)
    self.linear2 = nn.Linear(512, 512)
    self.linear3 = nn.Linear(512, 10)
    self.relu = nn.ReLU()
    self.softmax = nn.Softmax()

  def forward(self, x):
    y = self.linear1(x)
    y = self.relu(y)
    y = self.linear2(y)
    y = self.relu(y)
    y = self.linear3(y)
    y = self.softmax(y)
    return y

input = torch.randn(100, 784)
model = MLP()

torch.onnx.export(model, (input), "mnist_mlp.onnx", export_params=False)
