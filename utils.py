import collections
import datetime
import logging
import os
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils import data
from torchvision import datasets
from torchvision import transforms
import torchvision.models as models


def get_current_ts() -> str:
  return datetime.datetime.now().strftime("%Y%m%dT%H%M%S")


def get_path(cfg, key):
  root_path = cfg["path", "root"].format(experiment=cfg["experiment"])
  if key == "root":
    return root_path
  return os.path.join(root_path, cfg["path", key])


def load_model(in_channels: int, model_name: str = None) -> nn.Module:
  match model_name:
    case "resnet18":
      model = models.resnet18(num_classes=10)
    case "resnet50":
      model = models.resnet50(num_classes=10)
    case _:
      model = models.resnet18(num_classes=10)
  model.conv1 = nn.Conv2d(
      in_channels=in_channels,
      out_channels=model.conv1.weight.shape[0],
      kernel_size=3,
      stride=1,
      padding=1,
      bias=False,
  )
  model.maxpool = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = model.to(device)
  return model


class MutableMNIST(datasets.MNIST):

  def __init__(self, *args, **kwargs):
    super(MutableMNIST, self).__init__(*args, **kwargs)

  def __setitem__(self, index, val):
    img, target = val
    self.data[index] = img
    self.targets[index] = target

  def update_adv(self, index, target):
    self.data[index, 0, 0] = 1
    self.targets[index] = target


def modify_data(data, cfg):
  n_change = int(len(data) * cfg["adv", "percent"] / 100)

  target = cfg["adv", "target"]
  indices = torch.Tensor([1 if label != target else 0 for _, label in data])
  indices = indices.nonzero().squeeze(1)
  indices = indices[torch.randperm(indices.size(0))][:n_change].tolist()

  for idx in indices:
    data.update_adv(idx, target)


def load_data(cfg) -> Tuple[data.DataLoader, data.DataLoader, data.DataLoader]:
  transform = transforms.Compose([
      transforms.Pad(
          (cfg["data", "padded_img_size"] - cfg["data", "img_size"]) // 2),
      transforms.ToTensor(),
      transforms.Normalize(mean=(cfg["data", "norm_mean"],),
                           std=(cfg["data", "norm_std"],)),
  ])
  train_data = MutableMNIST(
      root="data",
      train=True,
      download=True,
      transform=transform,
  )
  test_data = MutableMNIST(
      root="data",
      train=False,
      download=True,
      transform=transform,
  )

  if cfg["adv", "flip"]:
    modify_data(train_data, cfg)
    modify_data(test_data, cfg)

  batch_size = cfg["train", "batch_size"]
  train_loader = data.DataLoader(
      dataset=train_data,
      batch_size=batch_size,
      shuffle=True,
      drop_last=True,
  )
  test_loader = data.DataLoader(
      dataset=test_data,
      batch_size=batch_size,
      shuffle=False,
      drop_last=True,
  )
  return train_loader, test_loader


class Metrics:

  def __init__(self, tb_writer) -> None:
    self.tb_writer = tb_writer

    self.acc = collections.defaultdict(list)
    self.loss = collections.defaultdict(list)

    # NC 1.
    self.wc_nc = collections.defaultdict(list)

    self.act_equi_norm = collections.defaultdict(list)
    self.w_equi_norm = collections.defaultdict(list)

    self.std_act_cos_c = collections.defaultdict(list)
    self.std_w_cos_c = collections.defaultdict(list)

    self.max_equi_angle_act = collections.defaultdict(list)
    self.max_equi_angle_w = collections.defaultdict(list)

    self.w_act = collections.defaultdict(list)
    self.ncc_mismatch = collections.defaultdict(list)

  def append_items(self, epoch: int, metric_type: str, **kwargs):
    logging.info(f"{metric_type.title()} " + "-" * 10)
    for key, value in kwargs.items():
      getattr(self, key)[metric_type].append(value)
      logging.info(f"{key}: {value:.6f}.")
      self.tb_writer.add_scalar(f"{key}/{metric_type}", value, epoch)


class Features:

  def __init__(self, module: nn.Module):
    self.hook = module.register_forward_hook(self.hook_fn)

  def hook_fn(self, module, inputs, outputs):
    self.value = inputs[0].detach().clone()


def get_accuracy(labels: torch.Tensor, outputs: torch.Tensor) -> float:
  pred_classes = torch.argmax(outputs, dim=1)
  acc = (pred_classes == labels).float().mean().item()
  return acc


def get_device() -> str:
  return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_off_diag_mask(size: int) -> torch.Tensor:
  return ~torch.eye(size, dtype=torch.bool, device=get_device())
