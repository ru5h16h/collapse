import logging
import os
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils import data

import utils

TIMESTAMP = "20240611T192306"
MODEL = "resnet18"


def evaluate(
    model: nn.Module,
    data_loader: data.DataLoader,
    loss_fn: nn.Module,
) -> Tuple[float, float]:
  running_loss = 0
  model.eval()
  correct_predictions = 0
  total_predictions = 0
  with torch.no_grad():
    for idx, val_data in enumerate(data_loader):
      inputs, labels = val_data
      outputs = model(inputs)
      loss = loss_fn(outputs, labels)

      _, predictions = torch.max(outputs, 1)
      correct_predictions += (predictions == labels).sum().item()
      total_predictions += labels.size(0)
      running_loss += loss

  avg_loss = running_loss / (idx + 1)
  acc = correct_predictions / total_predictions
  return avg_loss, acc


def main():
  model = utils.load_model(model_name=MODEL)

  train_loader, _, _ = utils.load_data(shuffle_train_set=False)

  loss_fn = nn.CrossEntropyLoss()

  model_dir = f"runs/mnist/{TIMESTAMP}"
  epochs = list(range(25))

  for epoch_idx in epochs:
    file_path = os.path.join(model_dir, f"model_{epoch_idx + 1}")
    if not os.path.isfile(file_path):
      logging.info(f"{file_path} invalid.")
      continue

    model.load_state_dict(torch.load(file_path))
    logging.info(f"{file_path} loaded.")
    model.eval()

    train_loss, train_acc = evaluate(model, train_loader, loss_fn)
    val_loss, val_acc = evaluate(model, train_loader, loss_fn)
    logging.info(f"Epoch {epoch_idx + 1}. "
                 f"Loss: Train {train_loss:0.6f}. Val {val_loss:0.6f} "
                 f"Accuracy: Train {train_acc:0.6f}. Val {val_acc:0.6f} ")


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)
  main()
