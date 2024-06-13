import datetime
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils import data
from torchvision import datasets
from torchvision import transforms
import torchvision.models as models

BATCH_SIZE = 16
SEED = 42


def get_current_ts() -> str:
  return datetime.datetime.now().strftime("%Y%m%dT%H%M%S")


def load_model(model_name: str = None) -> nn.Module:
  match model_name:
    case "resnet18":
      model = models.resnet18(num_classes=10)
    case "resnet50":
      model = models.resnet50(num_classes=10)
    case _:
      model = models.resnet18(num_classes=10)
  model.conv1 = nn.Conv2d(
      in_channels=1,
      out_channels=64,
      kernel_size=(7, 7),
      stride=(2, 2),
      padding=(3, 3),
      bias=False,
  )
  return model


def load_data(
    shuffle_train_set: bool = True
) -> Tuple[data.DataLoader, data.DataLoader, data.DataLoader]:
  torch.manual_seed(SEED)
  transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
  ])
  train_data = datasets.MNIST(
      root="data",
      train=True,
      download=True,
      transform=transform,
  )
  test_data = datasets.MNIST(
      root="data",
      train=False,
      download=True,
      transform=transform,
  )

  train_size = int(0.8 * len(train_data))
  val_size = len(train_data) - train_size
  train_data, val_data = data.random_split(train_data, [train_size, val_size])

  train_loader = data.DataLoader(
      train_data,
      batch_size=BATCH_SIZE,
      shuffle=shuffle_train_set,
  )
  val_loader = data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
  test_loader = data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
  return train_loader, val_loader, test_loader
