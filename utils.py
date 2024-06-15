import datetime
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

DEBUG = True


def get_current_ts() -> str:
  return datetime.datetime.now().strftime("%Y%m%dT%H%M%S")


def load_model(input_channels: int, model_name: str = None) -> nn.Module:
  match model_name:
    case "resnet18":
      model = models.resnet18(pretrained=False, num_classes=10)
    case "resnet50":
      model = models.resnet50(pretrained=False, num_classes=10)
    case _:
      model = models.resnet18(pretrained=False, num_classes=10)
  model.conv1 = nn.Conv2d(
      in_channels=input_channels,
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
  test_data = datasets.MNIST(
      root="data",
      train=False,
      download=True,
      transform=transform,
  )
  train_loader = data.DataLoader(
      train_data,
      batch_size=BATCH_SIZE,
      shuffle=True,
  )
  test_loader = data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
  return train_loader, test_loader


class Metrics:

  def __init__(self) -> None:
    self.acc = []
    self.loss = []
