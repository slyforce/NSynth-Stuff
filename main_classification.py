import torch
import os
import json
import numpy as np
import time

import argparse

from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

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
  parser.add_argument("--batch_size", type=int, default=128, help="")
  parser.add_argument("--evaluate_after_n_steps", type=int, default=100, help="")

  parser.add_argument("--gpu", action="store_true", help="")
  parser.add_argument("--num_data_workers", type=int, default=1, help="")

  parser.add_argument("--model_name", type=str, default="cnn",
                      help="Which model to use. ")

  parser.add_argument("--model_directory",
                      type=str, default="/tmp/model",
                      help="Where to save the model.")
  parser.add_argument("--load_model_from",
                      type=str, default="",
                      help="Where to save the model.")

  parser.add_argument("--command", type=str, choices=["train", "evaluate"],
                      default="train", help="Which function is going to be executed.")

  args = parser.parse_args()
  return args

ARGS = get_arguments()


def eval(model: torch.nn.Module, dataset: DataLoader):
  model.eval()

  acc_meter = nn.utils.AverageMeter("Accuracy")
  loss_meter = nn.utils.AverageMeter("Loss")
  loss_mod = torch.nn.NLLLoss()
  for x, y in dataset:
    if ARGS.gpu:
      x = x.cuda()
      y = y.cuda()
    B = x.size()[0]

    y_logprobs = model(x)
    loss = loss_mod(y_logprobs, y)
    num_hits = nn.losses.accuracy_num_hits(y_logprobs, y)

    acc_meter.update(num_hits, n=B)
    loss_meter.update(loss, n=B)

  print(f"Evaluation results: {acc_meter} {loss_meter}")


def train(model: torch.nn.Module, opt, train_dataset: DataLoader, validation_dataset: DataLoader):

  loss_meter = nn.utils.AverageMeter("Loss")
  acc_meter = nn.utils.AverageMeter("Accuracy")
  speed_meter = nn.utils.AverageMeter("s/step")

  loss_mod = torch.nn.NLLLoss()
  num_steps = 0 
  for epoch in range(100):
    model.train()
    for x, y in train_dataset:
      tstart = time.time()
      num_steps += 1

      if ARGS.gpu:
        x = x.cuda()
        y = y.cuda()

      opt.zero_grad()
      y_logprobs = model(x)

      acc_meter.update(nn.losses.accuracy(y_logprobs, y))

      loss = loss_mod(y_logprobs, y)
      loss.backward()
      opt.step()
      loss_meter.update(loss.item())
      speed_meter.update(time.time() - tstart)

      if num_steps % 10 == 0:
        print(f"epoch={epoch} step={num_steps} {loss_meter} {speed_meter} {acc_meter}")
      if num_steps % 100 == 0:
        print(f"Evaluating after {num_steps}")
        eval(model, validation_dataset)
        save_model(model, ARGS.model_directory, num_steps)


def save_model(model: torch.nn.Module, directory: str, step: int):
  fname = os.path.join(directory, f"params-{step}")
  print(f"Saving model to {fname}")
  torch.save(model.state_dict(), fname)


def load_model(model: torch.nn.Module, path: str):
  print(f"Loading model from {path}")
  model.load_state_dict(torch.load(path))


def command_evaluate():
  ds_train = data.NSynthClassificationDataset(ARGS.training_data, "instrument_family")
  ds_valid = data.NSynthClassificationDataset(ARGS.valid_data, "instrument_family")
  valid_dl = DataLoader(ds_valid, batch_size=ARGS.batch_size, num_workers=ARGS.num_data_workers)

  model = nn.model.create_model(ARGS.model_name, ds_train.num_classes())
  if ARGS.load_model_from:
    load_model(model, ARGS.load_model_from)
  nn.model.model_statistics(model)
  if ARGS.gpu:
    model = model.cuda()

  all_y_pred, all_y = [], []
  for x, y in valid_dl:
    model.eval()
    y_pred = model.forward(x)
    y_pred = torch.argmax(y_pred, dim=-1)
    all_y_pred.append(y_pred)
    all_y.append(y)

  y = torch.cat(all_y)
  y_pred = torch.cat(all_y_pred)

  accuracy = losses.accuracy(y_pred, y)
  p, r, _ , _ = precision_recall_fscore_support(y, y_pred)

  print(f"Precision: {p}")
  print(f"Recall: {r}")
  print(f"acc: {accuracy}")


def command_train():
  ds_train = data.NSynthClassificationDataset(ARGS.training_data, "instrument_family")
  ds_valid = data.NSynthClassificationDataset(ARGS.valid_data, "instrument_family")
  train_dl = DataLoader(ds_train, batch_size=ARGS.batch_size, shuffle=True, num_workers=ARGS.num_data_workers)
  valid_dl = DataLoader(ds_valid, batch_size=ARGS.batch_size, num_workers=ARGS.num_data_workers)

  model = nn.model.create_model(ARGS.model_name, ds_train.num_classes())
  utils.mkdir_p(ARGS.model_directory)
  if ARGS.load_model_from:
    load_model(model, ARGS.load_model_from)

  nn.model.model_statistics(model)
  if ARGS.gpu:
    model = model.cuda()

  opt = torch.optim.Adam(model.parameters(), lr=1e-3)
  train(model, opt, train_dl, valid_dl)


if __name__ == "__main__":
  if ARGS.command == "train":
    command_train()
  elif ARGS.command == "evaluate":
    command_evaluate()
