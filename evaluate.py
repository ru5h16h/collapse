import logging
import os
from typing import Tuple

import tqdm
import torch
import torch.nn as nn
from torch.utils import data

import utils

TIMESTAMP = "20240611T192306"
MODEL = "resnet18"


def evaluate(
    epoch_idx: int,
    model: nn.Module,
    device: str,
    data_loader: data.DataLoader,
    loss_fn_red: nn.Module,
    metrics: utils.Metrics,
) -> Tuple[float, float]:
  loss = 0
  net_cor = 0
  model.eval()
  data_len = len(data_loader)
  p_bar = tqdm.tqdm(total=data_len, position=0, leave=True)
  idx = 0
  with torch.no_grad():
    for inputs, labels in data_loader:
      if inputs.shape[0] != utils.BATCH_SIZE:
        continue

      inputs, labels = inputs.to(device), labels.to(device)
      outputs = model(inputs)

      net_cor += (torch.max(outputs, 1)[1] == labels).sum().item()
      loss += loss_fn_red(outputs, labels).item()
      p_bar.update(1)
      p_bar.set_description(
          f"Analysis. Epoch: {epoch_idx + 1} [{idx + 1}/{data_len}]")

      idx += 1
      if utils.DEBUG and idx == 20:
        break
  p_bar.close()
  metrics.loss.append(loss / (idx * utils.BATCH_SIZE))
  metrics.acc.append(net_cor / (idx * utils.BATCH_SIZE))


def main():
  model = utils.load_model(model_name=MODEL)

  train_loader, test_loader = utils.load_data(shuffle_train_set=False)

  loss_fn = nn.CrossEntropyLoss()

  model_dir = f"runs/{TIMESTAMP}"
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
    test_loss, test_acc = evaluate(model, test_loader, loss_fn)
    logging.info(f"Epoch {epoch_idx + 1}. "
                 f"Loss: Train {train_loss:0.6f}. Val {test_loss:0.6f} "
                 f"Accuracy: Train {train_acc:0.6f}. Val {test_acc:0.6f} ")


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)
  main()
