import torch
import os
import numpy as np

from torch.utils.data import DataLoader, Dataset
from nn import utils

class NSynthClassificationDataset(Dataset):
  def __init__(self,
               path: str,
               classification_key: str):
    self.path = path
    self.classification_key = classification_key
    self.json_data = utils.load_data(os.path.join(self.path, "examples.json"))
    print(f"I have read {len(self.json_data)} examples from {self.path}.")
    self.classes_to_samples = utils.groupby(self.json_data.values(),
                                            [d[classification_key] for d in self.json_data.values()])
    print("Statistics: ")
    for k, v in utils.compute_statistics(self.classes_to_samples).items():
      print(f" {k}: {v}")

    self.samples_as_list = []
    for class_sample, samples in self.classes_to_samples.items():
      self.samples_as_list += samples

  def num_classes(self):
    return len(self.classes_to_samples)

  def __len__(self):
    return len(self.samples_as_list)

  def __getitem__(self, item: int):
    sample = self.samples_as_list[item]
    audio_fname = os.path.join(self.path, "audio", f"{sample['note_str']}.wav")

    # read a file with 16-bit PCM values (signed 16-bit integers)
    with open(audio_fname, "rb") as reader:
      frames = reader.read()
      signal = np.array(np.frombuffer(frames, dtype=np.int16), dtype=np.float32)
      signal = signal / 2 ** 15  # not 16, because 1 bit is used for sign
      signal = signal.reshape([-1, 1])
    return signal, sample[self.classification_key]

