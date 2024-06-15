import logging
import os
from typing import Tuple

import numpy as np
from scipy.sparse import linalg
import tqdm
import torch
import torch.nn as nn
from torch.utils import data

import utils

TIMESTAMP = "20240611T192306"
MODEL = "resnet18"

N_CLASSES = 10


def get_class_means(
    data_loader: data.DataLoader,
    device: str,
    model: nn.Module,
    features: utils.Features,
):
  # For computing mean per class.
  mean = [0 for _ in range(N_CLASSES)]
  n_per_class = [0 for _ in range(N_CLASSES)]
  data_len = len(data_loader)
  p_bar = tqdm.tqdm(total=data_len, position=0, leave=True)
  idx = 0
  for inputs, labels in data_loader:
    if inputs.shape[0] != utils.BATCH_SIZE:
      continue
    inputs, labels = inputs.to(device), labels.to(device)
    model(inputs)
    hid = features.value.view(inputs.shape[0], -1)
    for cl in range(N_CLASSES):
      idxs = (labels == cl).nonzero(as_tuple=True)[0]
      if len(idxs) == 0:
        continue
      hid_cl = hid[idxs, :]
      mean[cl] += torch.sum(hid_cl, axis=0)
      n_per_class[cl] += hid_cl.shape[0]
    p_bar.update(1)
    p_bar.set_description(f"Mean [{idx + 1}/{data_len}]")
    idx += 1
    if utils.DEBUG and idx == 20:
      break
  for cl in range(N_CLASSES):
    mean[cl] /= (n_per_class[cl] + 1e-9)
  return mean


def get_within_class_cov_and_other(
    data_loader: data.DataLoader,
    device: str,
    model: nn.Module,
    features: utils.Features,
    mean: torch.Tensor,
    loss_fn_red: nn.Module,
):
  loss = 0
  acc = 0

  data_len = len(data_loader)
  p_bar = tqdm.tqdm(total=data_len, position=0, leave=True)
  idx = 0
  Sw = 0
  for inputs, labels in data_loader:
    if inputs.shape[0] != utils.BATCH_SIZE:
      continue
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs)
    acc += (torch.max(outputs, 1)[1] == labels).sum().item()
    loss += loss_fn_red(outputs, labels).item()

    hid = features.value.view(inputs.shape[0], -1)
    for cl in range(N_CLASSES):
      idxs = (labels == cl).nonzero(as_tuple=True)[0]
      if len(idxs) == 0:
        continue
      hid_cl = hid[idxs, :]

      hid_cl_ = hid_cl - mean[cl].unsqueeze(0)
      cov = torch.matmul(hid_cl_.unsqueeze(-1), hid_cl_.unsqueeze(1))
      Sw += torch.sum(cov, dim=0)

    p_bar.update(1)
    p_bar.set_description(f"Covariance [{idx + 1}/{data_len}]")
    idx += 1
    if utils.DEBUG and idx == 20:
      break

  n_samples = (idx + 1) * utils.BATCH_SIZE
  Sw /= n_samples
  loss /= n_samples
  acc /= n_samples
  return Sw, loss, acc


def evaluate(
    model: nn.Module,
    device: str,
    data_loader: data.DataLoader,
    loss_fn_red: nn.Module,
    metrics: utils.Metrics,
    features: utils.Features,
) -> Tuple[float, float]:
  # Get class means.
  mean = get_class_means(data_loader, device, model, features)
  # Compute global mean.
  mean_c = torch.stack(mean).T
  mean_g = torch.mean(mean_c, dim=1, keepdim=True)

  # Between-class covariance
  mean_c_ = mean_c - mean_g
  Sb = torch.matmul(mean_c_, mean_c_.T) / N_CLASSES

  # Get with-in class covariance.
  Sw, loss, acc = get_within_class_cov_and_other(data_loader, device, model,
                                                 features, mean, loss_fn_red)

  # tr{Sw Sb^-1}
  Sw = Sw.cpu().detach().numpy()
  Sb = Sb.cpu().detach().numpy()
  eigvec, eigval, _ = linalg.svds(Sb, k=N_CLASSES - 1)
  inv_Sb = eigvec @ np.diag(eigval**(-1)) @ eigvec.T

  metrics.loss.append(loss)
  metrics.acc.append(acc)
  metrics.Sw_invSb.append(np.trace(Sw @ inv_Sb))


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
