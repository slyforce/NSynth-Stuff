import torch
from torch import nn
from torch.nn import functional as F
from torchaudio.transforms import MFCC
from typing import *
import numpy as np

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

class ConvolutionalEncoder(nn.Module):
  def __init__(self, input_dim: int, hidden_dim: int):
    super().__init__()
    stride = 2
    kernel = 5
    num_layers = 4
    dropout = 0.2
    self.layers = nn.Sequential()
    for i in range(num_layers):
      _input_dim = input_dim if i == 0 else hidden_dim
      #self.layers.add_module(f"bn{i}", nn.BatchNorm1d(_input_dim))
      self.layers.add_module(f"dropout{i}", nn.Dropout(dropout))
      self.layers.add_module(f"conv{i}", nn.Conv1d(_input_dim, hidden_dim, kernel_size=(kernel, )))
      self.layers.add_module(f"maxpool{i}", nn.MaxPool1d(stride))
      self.layers.add_module(f"relu{i}", nn.ReLU())

    print(f"Going to reduce the dimension of the input by {stride**num_layers}")

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # x: shape [B, D, T]
    # convolutional layers expect [B, D, T]
    return self.layers(x)


class CNNClassifier(nn.Module):

  def __init__(self, num_output_classes: int, hidden_dim: int):
    super().__init__()
    dropout = 0.2
    self.audio_processing = AudioProcessing()
    self.classifier = MultiClassClassifier(hidden_dim, num_output_classes)
    self.conv_encoder = ConvolutionalEncoder(self.audio_processing.output_dim(), hidden_dim)
    self.mlp = nn.Sequential(
      nn.Dropout(dropout),
      nn.Linear(hidden_dim, hidden_dim),
      nn.ReLU(),
      nn.Dropout(dropout),
      nn.Linear(hidden_dim, hidden_dim),
      nn.ReLU())

  def forward(self, raw_audio_signal: torch.Tensor) -> torch.Tensor:

    #print(f"Raw audio signal: {raw_audio_signal.size()}")
    mfcc_features = self.audio_processing.forward(raw_audio_signal)
    #print(f"Raw audio signal: {mfcc_features.size()}")

    hidden_projections = self.conv_encoder.forward(mfcc_features)  # shape [B, D, T]
    hidden_state, _ = torch.max(hidden_projections, dim=-1)
    hidden_state = self.mlp(hidden_state)
    log_probs = self.classifier.forward(hidden_state)
    return log_probs


class CNNLSTMClassifier(nn.Module):
  def __init__(self, num_output_classes: int):
    super().__init__()
    hidden_dim = 64
    self.classifier = MultiClassClassifier(hidden_dim, num_output_classes)
    self.conv_encoder = ConvolutionalEncoder(hidden_dim)
    self.lstm_encoder = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim,
                                num_layers=2, batch_first=True)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.conv_encoder.forward(x) # shape [B, T, D]
    x, _ = self.lstm_encoder.forward(x)
    x = x[:,-1,:] # we just want the final state for the classification
    return self.classifier.forward(x)


def model_statistics(model: nn.Module):
  num_params = 0
  for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")
    num_params += np.prod(param.shape)
  print(f"Total number of parameters: {num_params}")


class AudioProcessing(nn.Module):
  def __init__(self):
    super().__init__()
    self.transform = MFCC(sample_rate=16_000)

  def output_dim(self):
    return self.transform.n_mfcc

  def forward(self, x):
    # x: shape [B, T, 1]
    x = x.squeeze(2)

    # expects [B, T]
    x = self.transform.forward(x)

    # [B, D, T]
    return x


def create_model(model_name: str, number_output_classes: int, hidden_dim: int):
  if model_name == "cnn-lstm":
    return CNNLSTMClassifier(number_output_classes)
  elif model_name == "cnn":
    return CNNClassifier(number_output_classes, hidden_dim)
  else:
    raise ValueError(f"No model found for {model_name}")


# x_1^T, x_t in [-1, 1]
# The problem is that T is 64k, we want it to be more like 200

