#!/usr/bin/env python3

import functools
import logging
import os

import tqdm
import torch
from torch import optim
import torch.nn as nn
from torch.utils import tensorboard
from torch.utils import data

import utils
import evaluate

MODEL = "resnet18"
EPOCHS = 350

WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
LEARNING_RATE = 0.0679

LR_DECAY = 0.1
EPOCHS_LR_DECAY = [EPOCHS // 3, EPOCHS * 2 // 3]

EPOCH_LIST = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 20, 22, 24,
    27, 29, 32, 35, 38, 42, 45, 50, 54, 59, 65, 71, 77, 85, 92, 101, 110, 121,
    132, 144, 158, 172, 188, 206, 225, 245, 268, 293, 320, 349
]


def train_epoch(
    epoch_idx: int,
    model: nn.Module,
    train_loader: data.DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
) -> float:
  # Set the training flag.
  model.train()
  device = utils.get_device()

  data_len = len(train_loader)
  p_bar = tqdm.tqdm(total=data_len, position=0, leave=True)
  for idx, (inputs, labels) in enumerate(train_loader):
    inputs, labels = inputs.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs = model(inputs)

    loss = loss_fn(outputs, labels)
    loss.backward()
    optimizer.step()

    acc = utils.get_accuracy(labels, outputs)

    p_bar.update(1)
    p_bar.set_description(
        f"Train. Epoch {epoch_idx}. [{idx + 1}/{data_len}] "
        f"Batch Loss: {loss.item():.6f}. Batch Accuracy: {acc:.6f}.")

    if utils.DEBUG and idx == 20:
      break
  p_bar.close()


def main():
  timestamp = utils.get_current_ts()
  logging.info(f"Timestamp: {timestamp}.")

  model = utils.load_model(model_name=MODEL, in_channels=utils.INPUT_CHANNELS)
  features = utils.Features(module=model.fc)

  train_loader, test_loader = utils.load_data()

  loss_fn = nn.CrossEntropyLoss()
  loss_fn_red = nn.CrossEntropyLoss(reduction='sum')

  optimizer = optim.SGD(
      params=model.parameters(),
      lr=LEARNING_RATE,
      momentum=MOMENTUM,
      weight_decay=WEIGHT_DECAY,
  )
  lr_scheduler = optim.lr_scheduler.MultiStepLR(
      optimizer=optimizer,
      milestones=EPOCHS_LR_DECAY,
      gamma=LR_DECAY,
  )

  os.makedirs(f"runs/{timestamp}")
  writer = tensorboard.writer.SummaryWriter(f"runs/{timestamp}/writer")

  metrics = utils.Metrics(tb_writer=writer)
  for epoch_idx in range(EPOCHS):
    train_epoch(
        epoch_idx=epoch_idx,
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
    )
    lr_scheduler.step()

    if epoch_idx in EPOCH_LIST:
      evaluate_p = functools.partial(
          evaluate.evaluate,
          epoch_idx=epoch_idx,
          model=model,
          loss_fn_red=loss_fn_red,
          metrics=metrics,
          features=features,
      )
      evaluate_p(data_loader=train_loader, loader_type="train")
      evaluate_p(data_loader=test_loader, loader_type="test")

      model_path = f"runs/{timestamp}/model_{epoch_idx}"
      torch.save(model.state_dict(), model_path)

    writer.flush()
    if utils.DEBUG and epoch_idx == 2:
      break
    logging.info("_" * 79 + "\n")


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)
  main()
