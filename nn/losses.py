import torch


def accuracy_num_hits(y_pred: torch.Tensor, y: torch.Tensor):
  # y_pred has shape [B, C] C = number of Classes
  # y has shape [B]
  y_pred = torch.argmax(y_pred, dim=-1) # shape: [B]
  return (y_pred == y).long().sum()

def accuracy(y_pred: torch.Tensor, y: torch.Tensor):
  # y_pred has shape [B, C] C = number of Classes
  # y has shape [B]
  num_hits = accuracy_num_hits(y_pred, y)
  num_samples = y.size()
  return num_hits / num_samples, num_hits, num_samples