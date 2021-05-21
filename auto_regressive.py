import json
import os
import wave
import numpy as np
from typing import *
import argparse
import tensorflow as tf
import tensorflow.keras as keras
import multiprocessing as mp
import time
import sys
from dataclasses import dataclass
import threading
from queue import Queue


sys.path.insert(0, os.path.dirname(__file__))

def get_arguments():
  parser = argparse.ArgumentParser()

  parser.add_argument("--training_data", type=str,
                      default="/mnt/data/tmpMiguel/nsynth-train", help="")
  parser.add_argument("--valid_data", type=str,
                      default="/mnt/data/tmpMiguel/nsynth-valid", help="")
  parser.add_argument("--test_data", type=str,
                      default="/mnt/data/tmpMiguel/nsynth-test", help="")

  parser.add_argument("--max_timesteps", type=int, default=64000, help="")
  parser.add_argument("--batch_size", type=int, default=32, help="")
  parser.add_argument("--evaluate_after_n_steps", type=int, default=100, help="")
  args = parser.parse_args()
  return args

ARGS = get_arguments()


qualities = {
  0: "bright",
  1: "dark",
  2: "distortion",
  3: "fast_decay",
  4: "long_release",
  5: "multiphonic",
  6: "nonlinear_env",
  7: "percussive",
  8: "reverb",
  9: "tempo-synced",
}

instrument_families = {
0: "bass",
1: "brass",
2: "flute",
3: "guitar",
4: "keyboard",
5: "mallet",
6: "organ",
7: "reed",
8: "string",
9: "synth_lead",
10: "vocal",
}

instrument_sources = {
0: "acoustic",
1: "electronic",
2: "synthetic",
}


@dataclass
class ModelInput:
  signals: np.array # [batch_size, T, 1]
  instrument: np.array    # [batch_size]
  family: np.array        # [batch_size]
  qualities: np.array     # [batch_size, 10]
  pitch: np.array         # [batch_size]
  velocity: np.array      # [batch_size]
  source: np.array        # [batch_size]

  # prediction [e1, e2, e3, e4] -> [e2, e3, e4, e5] (e5 does not exist, that's why we removed it)
  # labels: [e1, e2, e3, e4] (remove e1, because e1 is not going to be predicted)
  def input_sequence(self):
    return self.signals[:,:-1, :]

  def target_sequence(self):
    return self.signals[:,1:, :]


class DatasetReader:
  def __init__(self, path: str):
    self.path = path
    with open(os.path.join(self.path, "examples.json"), "r") as reader:
      self.json_data = json.load(reader)
    print(f"I have read {len(self.json_data)} examples from {self.path}.")

  def get_data(self, key: str):
    audio_fname = os.path.join(self.path, "audio", key + ".wav")
    data = self.json_data[key]

    # read a file with 16-bit PCM values (signed 16-bit integers)
    with open(audio_fname, "rb") as reader:
      frames = reader.read(2*ARGS.max_timesteps)
      signal = np.array(np.frombuffer(frames, dtype=np.int16), dtype=np.float32)
      signal = signal / 2 ** 15  # not 16, because 1 bit is used for sign
      signal = np.pad(signal, (1, 0)) # put a 0 at the beginning, but not at the end!

    return ModelInput(
      signals=signal.reshape([1, -1, 1]),
      instrument=np.array(data.get("instrument"), dtype=np.int32).reshape([1]),
      qualities=np.array(data.get("qualities"), dtype=np.int32).reshape([1, 10]),
      family=np.array(data.get("instrument_family"), dtype=np.int32).reshape([1]),
      pitch=np.array(data.get("pitch") - 21, dtype=np.int32).reshape([1]), # pitches are in range 21-108
      velocity=np.array(data.get("velocity"), dtype=np.int32).reshape([1]),
      source=np.array(data.get("instrument_source"), dtype=np.int32).reshape([1]),
    )

  def examples(self):
    for instrument_name, data in self.json_data.items():
      yield self.get_data(instrument_name)


class DatasetIterator:
  def __init__(self,
               reader: DatasetReader,
               shuffle: bool):
    self.reader = reader
    self.shuffle = shuffle
    self.example_keys = list(self.reader.json_data.keys())
    if self.shuffle:
      np.random.shuffle(self.example_keys)
    self._curr_idx = 0
    self._epochs = 0

    self.mp_pool = mp.Pool(processes=8)

  def __len__(self):
    return len(self.example_keys)

  def batch(self, batch_size: int):
    tstart = time.time()
    keys = self.example_keys[self._curr_idx:min(self._curr_idx+batch_size, len(self))]
    self._curr_idx += batch_size
    if self._curr_idx >= len(self):
      self._curr_idx -= len(self)
      self._epochs += 1
      if self.shuffle:
        np.random.shuffle(self.example_keys)

    model_input_list: Iterable[ModelInput] = self.mp_pool.map(self.reader.get_data, keys)
    audio_samples = np.zeros(shape=(batch_size, ARGS.max_timesteps+1, 1)) # dummy timestep at the beginning
    for i, data in enumerate(model_input_list):
      audio_samples[i, 0:, :] = data.signals[0, :ARGS.max_timesteps+1, :]

    model_input = ModelInput(
      signals=audio_samples,
      instrument=np.concatenate([m.instrument for m in model_input_list], axis=0), # [1,1] -> [32,1]
      qualities=np.concatenate([m.qualities for m in model_input_list], axis=0),
      family=np.concatenate([m.family for m in model_input_list], axis=0),
      pitch=np.concatenate([m.pitch for m in model_input_list], axis=0),
      velocity=np.concatenate([m.velocity for m in model_input_list], axis=0),
      source=np.concatenate([m.source for m in model_input_list], axis=0),
    )
    print(f"Time to fetch a {batch_size}-sample batch: {time.time()-tstart}")
    return model_input

  def iterate(self, batch_size):
    q = Queue(maxsize=30)

    def worker():
      original_num_epochs = self._epochs
      while original_num_epochs == self._epochs:
        q.put(self.batch(batch_size))

    thread = threading.Thread(target=worker)
    thread.daemon = True
    thread.start()

    while thread.is_alive():
      if q.empty():
        time.sleep(0.01)
      else:
        yield q.get()


class Electra(keras.Model):
  def __init__(self):
    super().__init__()

    self.lstm_signal = keras.layers.LSTM(64, return_sequences=True)

    self.output_projection = keras.layers.Dense(1, use_bias=True)

    self.family_embeddings = keras.layers.Embedding(len(instrument_families), 4)
    #self.quality_embeddings = keras.layers.Embedding(len(qualities), 4)
    self.source_embeddings = keras.layers.Embedding(len(instrument_sources), 4)
    self.pitch_embeddings = keras.layers.Embedding(88, 4)
    self.velocity_embeddings = keras.layers.Embedding(5, 4)
    self.instrument_embeddings = keras.layers.Embedding(1_006, 4)

    self.lstm_features = keras.layers.LSTM(64 + 4*5, return_sequences=True)


  def call(self, model_input: ModelInput, training=False):
    encoded_signal = self.lstm_signal(model_input.input_sequence())

    conditioning_features = keras.backend.concatenate(
      [
      self.family_embeddings(model_input.family),
      #self.quality_embeddings(model_input.qualities), # todo: this is somewhat difficult to handle, because this is not a one-hot encoding
      self.source_embeddings(model_input.source),
      self.pitch_embeddings(model_input.pitch),
      self.velocity_embeddings(model_input.velocity),
      self.instrument_embeddings(model_input.instrument),
      ]
    )
    final_lstm_input = keras.backend.concatenate([encoded_signal, conditioning_features])
    out = self.lstm_features(final_lstm_input)
    out = self.output_projection(out)
    return out


def l2_loss(outputs, labels):
  """
  :param outputs
    shape: [B,T,1]
  :param labels:
    shape: [B,T,1]
  """
  distances = tf.pow(outputs - labels, 2)
  distances = tf.reduce_sum(distances, axis=1)
  distances = tf.reduce_mean(distances, axis=0)
  return distances

class ExponentialMovingAverage:
  def __init__(self, beta=0.9):
    self.acc = 0.0
    self.calls = 0
    self.beta = beta

  def update(self, value):
    self.acc = self.beta * self.acc + (1.-self.beta) * value
    self.calls += 1
    return self.value()

  def value(self):
    return self.acc / (1-self.beta**self.calls)


def evaluate_on_dataset(model, iterator):

  loss = 0.0
  num_samples = 0
  for batch_idx, inputs in enumerate(iterator.iterate(ARGS.batch_size)):
    out = model(inputs, training=False)
    loss += l2_loss(out, inputs.target_sequence())
    num_samples += inputs.signals.shape[0]

  print("Evaluation done:")
  print(f" l2-loss {loss/num_samples:.4f}")
  print(f" #samples {num_samples}")


if __name__ == "__main__":
  model = Electra()
  opt = tf.keras.optimizers.Adam(learning_rate=1e-4)

  train_iterator = DatasetIterator(DatasetReader(ARGS.training_data), shuffle=True)
  valid_iterator = DatasetIterator(DatasetReader(ARGS.valid_data), shuffle=False)
  test_iterator = DatasetIterator(DatasetReader(ARGS.test_data), shuffle=False)

  total_step_time = 0
  total_steps = 0
  smoothed_loss = ExponentialMovingAverage()
  for epoch_idx in range(10):
    print(f"Starting epoch {epoch_idx+1}")

    for batch_idx, signals in enumerate(train_iterator.iterate(ARGS.batch_size)):
      # prediction [e1, e2, e3, e4] -> [e2, e3, e4, e5] (e5 does not exist, that's why we removed it)
      # labels: [e1, e2, e3, e4] (remove e1, because e1 is not going to be predicted)

      step_start = time.time()
      with tf.GradientTape() as tape:
        out = model(signals, training=True)
        if total_steps == 0:
          print("Model parameters: ")
          for weight in model.trainable_weights:
            print(f"{weight.name} shape: {weight.shape}")

        loss = l2_loss(out, signals.target_sequence())
        smoothed_loss.update(float(loss))
      grads = tape.gradient(loss, model.trainable_weights)
      opt.apply_gradients(zip(grads, model.trainable_weights))

      total_step_time += time.time() - step_start
      total_steps += 1

      if total_steps % 10 == 0:
        print(f"Step {total_steps}: loss={float(loss):.4f} "
              f"(smoothed={smoothed_loss.value():.4f}) "
              f"steps/sec={total_steps / total_step_time:.2f}")

      if total_steps % ARGS.evaluate_after_n_steps == 0:
        evaluate_on_dataset(model, valid_iterator)

