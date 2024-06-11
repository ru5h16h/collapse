#!/usr/bin/env python3

import datetime
import logging
import os
from typing import Tuple

import torch
from torch import optim
import torch.nn as nn
from torch.utils import tensorboard
from torch.utils import data
from torchvision import datasets
from torchvision import transforms
import torchvision.models as models

EPOCHS = 25
BATCH_SIZE = 16


def get_current_ts() -> str:
  return datetime.datetime.now().strftime("%Y%m%dT%H%M%S")


def load_model() -> nn.Module:
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


def load_data() -> Tuple[data.DataLoader, data.DataLoader, data.DataLoader]:
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
      shuffle=True,
  )
  val_loader = data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
  test_loader = data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
  return train_loader, val_loader, test_loader


def train_epoch(
    epoch_idx: int,
    model: nn.Module,
    train_loader: data.DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    tb_writer: tensorboard.writer.SummaryWriter,
) -> float:
  running_loss = 0
  for idx, data in enumerate(train_loader):
    inputs, labels = data
    optimizer.zero_grad()
    outputs = model(inputs)

    loss = loss_fn(outputs, labels)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
    if idx % 100 == 99 and idx != 0:
      avg_loss = running_loss / (idx + 1)
      iteration = epoch_idx * len(train_loader) + idx + 1
      logging.info(f"Step: {iteration}. Loss: {avg_loss:0.3f}.")
      tb_writer.add_scalar("Loss/train", avg_loss, iteration)
      running_loss = 0


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
  timestamp = get_current_ts()

  model = load_model()

  train_loader, val_loader, _ = load_data()

  loss_fn = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

  os.makedirs(f"runs/mnist/{timestamp}")
  writer = tensorboard.writer.SummaryWriter(f"runs/mnist/{timestamp}/writer")

  # best_val_loss = 100000
  for epoch_idx in range(EPOCHS):

    model.train(True)
    train_epoch(
        epoch_idx=epoch_idx,
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        tb_writer=writer,
    )

    train_loss, train_acc = evaluate(model, train_loader, loss_fn)
    val_loss, val_acc = evaluate(model, val_loader, loss_fn)

    logging.info(f"Epoch {epoch_idx + 1}. "
                 f"Loss: Train {train_loss:0.3f}. Val {val_loss:0.3f} "
                 f"Accuracy: Train {train_acc:0.3f}. Val {val_acc:0.3f} ")

    writer.add_scalars(
        main_tag="Training vs. Validation loss",
        tag_scalar_dict={
            "Training": train_loss,
            "Validation": val_loss
        },
        global_step=epoch_idx + 1,
    )
    writer.flush()

    # if val_loss < best_val_loss:
    # best_val_loss = val_loss
    model_path = f"runs/mnist/{timestamp}/model_{epoch_idx + 1}"
    torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)
  main()
