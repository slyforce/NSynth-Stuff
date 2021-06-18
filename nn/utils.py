import json
import numpy as np
from collections import defaultdict
from typing import *
import os
import errno


def groupby(data: Iterable[Any], keys: Iterable[Any]) -> Dict[Any, List[Any]]:
  keys_to_data = defaultdict(list)
  for d, k in zip(data, keys):
    keys_to_data[k].append(d)
  return keys_to_data


def load_data(fname: str) -> Union[Dict, List]:
  with open(fname, "rt") as reader:
    return json.load(reader)


def compute_statistics(grouped_dictionary: Dict[Any, List[Any]]) -> Dict[str, float]:
  lengths = np.array([len(x) for x in grouped_dictionary.values()])
  return {
    "mean": lengths.mean(),
    "std": lengths.std(),
    "min": lengths.min(),
    "max": lengths.max(),
    "num_classes": lengths.size,
    "num_samples": lengths.sum(),
  }

class AverageMeter(object):
  """Computes and stores the average and current value"""

  def __init__(self, name, fmt=':f'):
    self.name = name
    self.fmt = fmt
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

  def __str__(self):
    fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
    ret = fmtstr.format(**self.__dict__)
    return ret


def mkdir_p(path: str) -> None:
  """Make sure a directory exists (like mkdir -p on the shell)"""
  try:
    os.makedirs(path)
  except OSError as exc:
    if exc.errno == errno.EEXIST and os.path.isdir(path):
      pass
    else:
      raise

