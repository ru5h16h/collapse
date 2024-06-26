from typing import Tuple

import numpy as np
from scipy.sparse import linalg
import tqdm
import torch
import torch.nn as nn
from torch.utils import data

import utils


def get_class_means(
    data_loader: data.DataLoader,
    model: nn.Module,
    features: utils.Features,
    cfg,
) -> torch.Tensor:
  device = utils.get_device()
  n_classes = cfg["data", "n_classes"]
  target_cl = cfg["sub", "target"]

  sub_mean = 0
  sub_ctr = 0
  mean = torch.zeros(n_classes, features.value.shape[1], device=device)
  n_per_class = torch.zeros(n_classes, device=device)

  data_len = len(data_loader)
  p_bar = tqdm.tqdm(total=data_len, position=0, leave=True)

  with torch.no_grad():
    for idx, (inputs, labels) in enumerate(data_loader):
      inputs, labels = inputs.to(device), labels.to(device)
      outputs = model(inputs)
      prob = nn.functional.softmax(outputs, dim=1)
      hid = features.value.view(inputs.shape[0], -1)

      for cl in range(n_classes):
        idxs = (labels == cl).nonzero(as_tuple=True)[0]
        if len(idxs) == 0:
          continue
        hid_cl = hid[idxs, :]
        mean[cl] += torch.sum(hid_cl, axis=0)
        n_per_class[cl] += hid_cl.shape[0]

      sig_idxs = (labels != target_cl).nonzero(as_tuple=True)[0]
      prob_sig = prob[sig_idxs]
      sub_ctr += prob_sig.size(0)
      sub_mean_t = torch.max(prob_sig, dim=1)[0] - prob_sig[:, target_cl]
      sub_mean += sub_mean_t.sum().item()

      p_bar.update(1)
      p_bar.set_description(f"{'Mean':<10} [{idx + 1}/{data_len}]")

      if cfg["args", "debug"] and idx == 20:
        break

  mean /= n_per_class.unsqueeze(dim=1)
  sub_mean /= sub_ctr
  return mean.T, sub_mean


def get_within_class_cov_and_other(
    data_loader: data.DataLoader,
    model: nn.Module,
    features: utils.Features,
    mean_per_class: torch.Tensor,
    sub_mean: float,
    loss_fn_red: nn.Module,
    cfg,
):
  device = utils.get_device()
  n_classes = cfg["data", "n_classes"]

  loss = 0
  acc = 0
  ncc_mismatch = 0
  Sw = 0

  target_cl = cfg["sub", "target"]
  sub_ctr = 0
  sub_std = 0
  sub_op = [[] for _ in range(n_classes)]

  data_len = len(data_loader)
  p_bar = tqdm.tqdm(total=data_len, position=0, leave=True)

  with torch.no_grad():
    for idx, (inputs, labels) in enumerate(data_loader):
      inputs, labels = inputs.to(device), labels.to(device)
      outputs = model(inputs)
      pred = torch.argmax(outputs, dim=1)
      prob = nn.functional.softmax(outputs, dim=1)

      acc += (pred == labels).float().sum().item()
      loss += loss_fn_red(outputs, labels).item()

      hid = features.value.view(inputs.shape[0], -1)
      for cl in range(n_classes):
        idxs = (labels == cl).nonzero(as_tuple=True)[0]
        if len(idxs) == 0:
          continue
        hid_cl = hid[idxs, :]

        hid_cl_ = hid_cl - mean_per_class[cl].unsqueeze(0)
        cov = torch.matmul(hid_cl_.unsqueeze(-1), hid_cl_.unsqueeze(1))
        Sw += torch.sum(cov, dim=0)

        if target_cl != cl:
          req_prob = prob[idxs[:5]]
          sub_op_cl = torch.max(req_prob, dim=1)[0] - req_prob[:, target_cl]
          sub_op[cl] = sub_op_cl.tolist()

      sig_idxs = (labels != target_cl).nonzero(as_tuple=True)[0]
      prob_sig = prob[sig_idxs]
      sub_ctr += prob_sig.size(0)
      sub_std_t = torch.max(prob_sig, dim=1)[0] - prob_sig[:, target_cl]
      sub_std += torch.pow(sub_std_t - sub_mean, 2).sum().item()

      # Classifier behaviours approaches that of NCC. See figure 7.
      ncc_pred = (hid[:, None] - mean_per_class).norm(dim=2).argmin(dim=1)
      ncc_mismatch += (ncc_pred != pred).float().sum().item()

      p_bar.update(1)
      p_bar.set_description(f"{'Covariance':<10} [{idx + 1}/{data_len}]")
      if cfg["args", "debug"] and idx == 20:
        break

  n_samples = (idx + 1) * cfg["train", "batch_size"]
  Sw /= n_samples
  loss /= n_samples
  acc /= n_samples
  ncc_mismatch /= n_samples
  sub_std = (sub_std / sub_ctr)**(1 / 2)
  return Sw, loss, acc, ncc_mismatch, sub_op, sub_std


def evaluate(
    epoch_idx: int,
    model: nn.Module,
    data_loader: data.DataLoader,
    loader_type: str,
    loss_fn_red: nn.Module,
    metrics: utils.Metrics,
    features: utils.Features,
    cfg,
) -> Tuple[float, float]:
  # Set the eval flag.
  model.eval()

  metrics_d = {}
  n_classes = cfg["data", "n_classes"]

  # Get class means.
  mu_c, sub_mean = get_class_means(
      data_loader=data_loader,
      model=model,
      features=features,
      cfg=cfg,
  )
  # Compute global mean.
  mu_g = torch.mean(mu_c, dim=1, keepdim=True)

  # Between-class covariance
  mu_c_zm = mu_c - mu_g
  cov_bc = torch.matmul(mu_c_zm, mu_c_zm.T) / n_classes

  # Get with-in class covariance.
  (cov_wc, loss, acc, ncc_mismatch, sub_op,
   sub_std) = get_within_class_cov_and_other(
       data_loader=data_loader,
       model=model,
       features=features,
       mean_per_class=mu_c.T,
       sub_mean=sub_mean,
       loss_fn_red=loss_fn_red,
       cfg=cfg,
   )
  metrics_d["loss"] = loss
  metrics_d["acc"] = acc
  metrics_d["ncc_mismatch"] = ncc_mismatch
  metrics_d["sub_op"] = sub_op
  metrics_d["sub_mean"] = sub_mean
  metrics_d["sub_std"] = sub_std

  w_fc = model.fc.weight.T

  # # tr{Sw Sb^-1}. Training within-class variation collapse. See figure 6.
  cov_wc = cov_wc.cpu().detach().numpy()
  cov_bc = cov_bc.cpu().detach().numpy()
  eig_vec, eig_val, _ = linalg.svds(cov_bc, k=n_classes - 1)
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
  angle = 1 / (n_classes - 1)
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
  metrics.append_items(epoch_idx, loader_type, **metrics_d)
  metrics_path = utils.get_path(cfg, "metrics")
  metrics_path = metrics_path.format(set=loader_type, epoch_idx=epoch_idx)
  utils.write_pickle(metrics_path, metrics_d)
