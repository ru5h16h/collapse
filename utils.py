import datetime
import logging
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils import data
from torchvision import datasets
from torchvision import transforms
import torchvision.models as models

BATCH_SIZE = 128
SEED = 42
IMG_SIZE = 28
PADDED_IMG_SIZE = 32
INPUT_CHANNELS = 1

DEBUG = False


def get_current_ts() -> str:
  return datetime.datetime.now().strftime("%Y%m%dT%H%M%S")


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


def load_data() -> Tuple[data.DataLoader, data.DataLoader, data.DataLoader]:
  transform = transforms.Compose([
      transforms.Pad((PADDED_IMG_SIZE - IMG_SIZE) // 2),
      transforms.ToTensor(),
      transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
  ])
  train_data = datasets.MNIST(
      root="data",
      train=True,
      download=True,
      transform=transform,
  )
  train_data = data.Subset(train_data, list(range(2560)))
  test_data = datasets.MNIST(
      root="data",
      train=False,
      download=True,
      transform=transform,
  )
  train_loader = data.DataLoader(
      dataset=train_data,
      batch_size=BATCH_SIZE,
      shuffle=True,
      drop_last=True,
  )
  test_loader = data.DataLoader(
      dataset=test_data,
      batch_size=BATCH_SIZE,
      shuffle=False,
      drop_last=True,
  )
  return train_loader, test_loader


class Metrics:

  def __init__(self, tb_writer) -> None:
    self.tb_writer = tb_writer

    self.acc = []
    self.loss = []

    # NC 1.
    self.wc_nc = []

    self.act_equi_norm = []
    self.w_equi_norm = []

    self.std_act_cos_c = []
    self.std_w_cos_c = []

    self.max_equi_angle_act = []
    self.max_equi_angle_w = []

    self.w_act = []

  def append_items(self, epoch, **kwargs):
    for key, value in kwargs.items():
      getattr(self, key).append(value)
      logging.info(f"{key}: {value:.6f}.")
      self.tb_writer.add_scalar(key, value, epoch)


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
