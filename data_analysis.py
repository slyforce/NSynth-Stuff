import json
import numpy as np
from collections import defaultdict
from typing import *

def groupby(data: List[Any], keys: List[Any]): 
  keys_to_data = defaultdict(list)
  for d, k in zip(data, keys):
    keys_to_data[k].append(d)
  return keys_to_data

def load_data(fname: str):
  with open(fname, "rt") as reader:
    return json.load(reader)

# 7 7 0 81 43 01 84 1
# Apple Banana PassionFruit


class Fruit:
  name: str
  popularity: float

# compare(a, b) -> is_equal, is_smaller, is_greater
# comparison_value(Fruit) -> Fruit.popularity (since this is a float it has a compare(a,b) function for it)






def run(data: Dict[str, Dict]):
  print(f"# samples: {len(data)}")
  for key in ["instrument", "instrument_family", "instrument_source"]:
    print(f"Statistics for {key}")
    data_grouped_by_instrument = groupby(data.values(), [d[key] for d in data.values()]) # Dict[str, List[Any])
    data_grouped_by_instrument = dict(sorted(data_grouped_by_instrument.items(), key=lambda x: len(x[1])))
    num_samples_per_key = [len(x) for x in data_grouped_by_instrument.values()]

    print(f"Mean={np.mean(num_samples_per_key)} std={np.std(num_samples_per_key)} "
          f"min={np.min(num_samples_per_key)} max={np.max(num_samples_per_key)} "
          f"median={np.median(num_samples_per_key)} num_classes={len(data_grouped_by_instrument)}")
    #for instrument, samples in data_grouped_by_instrument.items():
    #  print(f"{samples[0]['instrument_str']} -> {len(samples)} ")


train_data = load_data("./nsynth-train/examples.json")
valid_data = load_data("./nsynth-valid/examples.json")
test_data = load_data("./nsynth-test/examples.json")

def check_that_training_keys_are_present_in_test_data():
  train_instruments = set(d["instrument_family"] for d in train_data.values())
  valid_instruments = set(d["instrument_family"] for d in valid_data.values())

  difference = valid_instruments.difference(train_instruments)
  print(f"Difference between train and validation set classes: {len(difference)} ")

run(train_data)
#check_that_training_keys_are_present_in_test_data()

# max y' p(y'|x)     y' is a class index, x is an input sample
# we need to access this model: p(y|x)
# p(y|x) = p(y) * p(x|y) / p(x) -> Bayes theorem


# max y' p(y'|x)
# = max y' p(y') * p(x|y') / p(x)
# = max y' p(y') * p(x|y')

# f(x) = Wx + b
