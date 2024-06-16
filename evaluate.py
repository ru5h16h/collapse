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
    model: nn.Module,
    features: utils.Features,
) -> torch.Tensor:
  device = utils.get_device()
  # For computing mean per class.
  mean = [0 for _ in range(N_CLASSES)]
  n_per_class = [0 for _ in range(N_CLASSES)]
  data_len = len(data_loader)
  p_bar = tqdm.tqdm(total=data_len, position=0, leave=True)
  idx = 0
  for inputs, labels in data_loader:
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
    model: nn.Module,
    features: utils.Features,
    mean_per_class: torch.Tensor,
    loss_fn_red: nn.Module,
):
  device = utils.get_device()
  loss = 0
  acc = 0

  data_len = len(data_loader)
  p_bar = tqdm.tqdm(total=data_len, position=0, leave=True)
  idx = 0
  Sw = 0
  for inputs, labels in data_loader:
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

      hid_cl_ = hid_cl - mean_per_class[cl].unsqueeze(0)
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
    epoch_idx: int,
    model: nn.Module,
    data_loader: data.DataLoader,
    loss_fn_red: nn.Module,
    metrics: utils.Metrics,
    features: utils.Features,
) -> Tuple[float, float]:
  # Set the eval flag.
  model.eval()

  metrics_d = {}

  # Get class means.
  mean_per_class = get_class_means(data_loader, model, features)
  # Compute global mean.
  mu_c = torch.stack(mean_per_class).T
  mu_g = torch.mean(mu_c, dim=1, keepdim=True)

  # Between-class covariance
  mu_c_zm = mu_c - mu_g
  cov_bc = torch.matmul(mu_c_zm, mu_c_zm.T) / N_CLASSES

  # Get with-in class covariance.
  cov_wc, loss, acc = get_within_class_cov_and_other(
      data_loader,
      model,
      features,
      mean_per_class,
      loss_fn_red,
  )
  metrics_d["loss"] = loss
  metrics_d["acc"] = acc

  w_fc = model.fc.weight.T

  # # tr{Sw Sb^-1}. Training within-class variation collapse. See figure 6.
  cov_wc = cov_wc.cpu().detach().numpy()
  cov_bc = cov_bc.cpu().detach().numpy()
  eig_vec, eig_val, _ = linalg.svds(cov_bc, k=N_CLASSES - 1)
  inv_cov_bc = eig_vec @ np.diag(eig_val**(-1)) @ eig_vec.T
  metrics_d["wc_nc"] = np.trace(cov_wc @ inv_cov_bc)

  # Train class mean becomes equinorm. See figure 2.
  # Last layer activations.
  norm_mu_c_zm = torch.norm(mu_c_zm, dim=0)
  act_equi_norm = (norm_mu_c_zm.std() / norm_mu_c_zm.mean()).item()
  metrics_d["act_equi_norm"] = act_equi_norm
  # Last layer classifier weights.
  norm_w_fc = torch.norm(w_fc, dim=0)
  w_equi_norm = (norm_w_fc.std() / norm_w_fc.mean()).item()
  metrics_d["w_equi_norm"] = w_equi_norm

  # Train class mean approaches equiangularity. See figure 3.
  mask = utils.get_off_diag_mask(mu_c_zm.size(1))
  # Last layer activations.
  mu_c_zm_norm = mu_c_zm / norm_mu_c_zm
  mu_c_zm_cos = mu_c_zm_norm.T @ mu_c_zm_norm
  metrics_d["std_act_cos_c"] = mu_c_zm_cos[mask].std().item()
  # Last layer classifier weights.
  w_fc_norm = w_fc / norm_w_fc
  w_fc_cos = w_fc_norm.T @ w_fc_norm
  metrics_d["std_w_cos_c"] = w_fc_cos[mask].std().item()

  # Train class mean approches maximal-angle equiangularity. See figure 4.
  angle = 1 / (N_CLASSES - 1)
  # Last layer activations.
  max_equi_angle_act = (mu_c_zm_cos[mask] + angle).abs().mean().item()
  metrics_d["max_equi_angle_act"] = max_equi_angle_act
  # Last layer classifier weights.
  max_equi_angle_w = (w_fc_cos[mask] + angle).abs().mean().item()
  metrics_d["max_equi_angle_w"] = max_equi_angle_w

  # Classifier converges to train class means. See figure 5.
  mu_c_zm_f_norm = mu_c_zm / torch.norm(mu_c_zm)
  w_fc_f_norm = w_fc / torch.norm(w_fc)
  metrics_d["w_act"] = torch.norm(mu_c_zm_f_norm - w_fc_f_norm).item()

  # Keep track of metrics.
  metrics.append_items(epoch_idx, **metrics_d)
