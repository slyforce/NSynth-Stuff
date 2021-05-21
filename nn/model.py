import torch
from torch import nn
from torch.nn import functional as F


class MultiClassClassifier(nn.Module):
  def __init__(self, input_dim: int, num_output_classes: int):
    super().__init__()
    self.proj = nn.Linear(input_dim, num_output_classes)

  def forward(self, x):
    logits = self.proj(x)
    logprobs = F.log_softmax(logits, dim=-1)
    return logprobs


class LSTMEncoder(nn.Module):
  def __init__(self, output_dimension: int):
    super(LSTMEncoder, self).__init__()
    self.initial_proj = nn.Linear(1, 16)
    self.lstm = nn.LSTM(input_size=16,
                        hidden_size=output_dimension,
                        num_layers=2, batch_first=True)

  def forward(self, signal):
    # signal: shape [B, T, 1]
    # B = Batch size
    # T = number of Timesteps
    # D = model Dimension

    x = F.relu(self.initial_proj(signal))
    x, state = self.lstm(x) # [B,T,D]
    return x


class LSTMClassifier(nn.Module):
  def __init__(self, num_output_classes: int):
    super().__init__()

    hidden_dim = 64
    self.classifier = MultiClassClassifier(hidden_dim, num_output_classes)
    self.encoder = LSTMEncoder(hidden_dim)

  def forward(self, x):
    x = self.encoder(x)[:,-1,:] # we just want the final state for the classification
    return self.classifier(x)