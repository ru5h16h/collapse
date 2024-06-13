import collections
import logging
import os
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from torchvision.models import feature_extraction
import tqdm

import utils

TIMESTAMP = "20240611T192306"
MODEL = "resnet18"
EPOCH = 25


def get_intermediate_model(model: nn.Module, layers: Dict[str, str]):
  int_model = feature_extraction.create_feature_extractor(
      model=model,
      return_nodes=layers,
  )
  return int_model


def get_mean(model, data_loader: data.DataLoader, layers):
  layer_model = get_intermediate_model(model, layers)
  means = {layer: collections.defaultdict(list) for layer in layers}
  for data in tqdm.tqdm(data_loader, total=len(data_loader), ncols=79):
    inputs, labels = data
    layer_outputs = layer_model(inputs)
    for layer in layers:
      layer_output = torch.squeeze(layer_outputs[layer])
      for idx, label in enumerate(labels.numpy()):
        means[layer][label].append(layer_output.detach().numpy()[idx])
  for layer in layers:
    for label in means[layer]:
      means[layer][label] = np.array(means[layer][label])
  return means


def get_unique_labels(data_loader: data.DataLoader):
  all_labels = []
  for _, labels in data_loader:
    all_labels.extend(labels.tolist())
  unique_labels = list(set(all_labels))
  return unique_labels


def main():
  train_loader, _ = utils.load_data()

  model = utils.load_model(model_name=MODEL)
  model_dir = f"runs/mnist/{TIMESTAMP}"
  file_path = os.path.join(model_dir, f"model_{EPOCH}")
  model.load_state_dict(torch.load(file_path))
  model.eval()

  layers = {"avgpool": "avgpool", "fc": "fc"}
  means = get_mean(model, train_loader, layers)

  for layer in layers:
    for label in means[layer]:
      logging.info(
          f"Class: {label}, Layer: {layer}, Mean: {means[layer][label]}")


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)
  main()
