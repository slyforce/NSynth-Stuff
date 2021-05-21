import torch
import os
import json
import numpy as np
import time

import argparse

from torch.utils.data import DataLoader, Dataset

import nn.model
from nn import utils
from nn import data
from nn import losses


def get_arguments():
  parser = argparse.ArgumentParser()

  parser.add_argument("--training_data", type=str,
                      default="./nsynth-train", help="")
  parser.add_argument("--valid_data", type=str,
                      default="./nsynth-valid", help="")
  parser.add_argument("--test_data", type=str,
                      default="./nsynth-test", help="")

  parser.add_argument("--max_timesteps", type=int, default=64000, help="")
  parser.add_argument("--batch_size", type=int, default=32, help="")
  parser.add_argument("--evaluate_after_n_steps", type=int, default=100, help="")
  args = parser.parse_args()
  return args

ARGS = get_arguments()


def eval(model: torch.nn.Module, dataset: DataLoader):
  model.eval()

  acc_meter = nn.utils.AverageMeter("Accuracy")
  loss_meter = nn.utils.AverageMeter("Loss")
  loss_mod = torch.nn.NLLLoss(reduction="sum")
  for x, y in dataset:
    B = x.shape()[0] # current batch size

    y_logprobs = model(x)
    loss = loss_mod(y_logprobs, y)
    num_hits = nn.losses.accuracy(y_logprobs, y)

    acc_meter.update(num_hits, n=B)
    loss_meter.update(loss, n=B)

  print(f"Evaluation results: {acc_meter} {loss_meter}")


def train(model: torch.nn.Module, opt, train_dataset: DataLoader, validation_dataset: DataLoader):

  loss_meter = nn.utils.AverageMeter("Loss")
  speed_meter = nn.utils.AverageMeter("s/step")

  loss_mod = torch.nn.NLLLoss()
  for epoch in range(100):
    model.train()
    for x, y in train_dataset:
      tstart = time.time()
      opt.zero_grad()
      y_logprobs = model(x)

      loss = loss_mod(y_logprobs, y)
      loss.backward()
      opt.step()
      loss_meter.update(loss.item())
      speed_meter.update(time.time() - tstart)
    eval(model, validation_dataset)


def main():
  ds_train = data.NSynthClassificationDataset(ARGS.training_data, "instrument_family")
  ds_valid = data.NSynthClassificationDataset(ARGS.valid_data, "instrument_family")
  train_dl = DataLoader(ds_train, batch_size=ARGS.batch_size)
  valid_dl = DataLoader(ds_valid, batch_size=ARGS.batch_size)

  model = nn.model.LSTMClassifier(ds_train.num_classes())
  opt = torch.optim.Adam(model.parameters(), lr=1e-3)

  train(model, opt, train_dl, valid_dl)


if __name__ == "__main__":
  main()
