#!/usr/bin/env python3

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


def train_epoch(
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
        f"Train. [{idx + 1}/{data_len}] "
        f"Batch Loss: {loss.item():.6f}. Batch Accuracy: {acc:.6f}.")

    if utils.DEBUG and idx == 20:
      break
  p_bar.close()


def main():
  timestamp = utils.get_current_ts()
  logging.info(f"Timestamp: {timestamp}.")

  model = utils.load_model(model_name=MODEL, in_channels=utils.INPUT_CHANNELS)
  features = utils.Features(module=model.fc)

  train_loader, _ = utils.load_data()

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

  metrics = utils.Metrics()
  for epoch_idx in range(EPOCHS):
    train_epoch(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
    )
    lr_scheduler.step()
    evaluate.evaluate(
        model=model,
        data_loader=train_loader,
        loss_fn_red=loss_fn_red,
        metrics=metrics,
        features=features,
    )
    logging.info(
        f"Epoch {epoch_idx + 1}.\n"
        f"Loss: {metrics.loss[-1]:.6f}.\n"
        f"Acc: {metrics.acc[-1]:.6f}.\n"
        f"With-in class variation collapse: {metrics.wc_nc[-1]:.6f}.\n"
        f"Class mean becomes equinorm: {metrics.act_equi_norm[-1]:.6f}.\n"
        f"Class mean approaches equiangularity: {metrics.std_cos_c[-1]:.6f}.\n"
        f"Class mean approach maximal-angle equiangularity: {metrics.max_equi_angle[-1]:.6f}."
    )

    writer.add_scalar("Loss", metrics.loss[-1], epoch_idx + 1)
    writer.add_scalar("Accuracy", metrics.acc[-1], epoch_idx + 1)
    writer.add_scalar("wc_nc", metrics.wc_nc[-1], epoch_idx + 1)
    writer.add_scalar("act_equi_norm", metrics.act_equi_norm[-1], epoch_idx + 1)
    writer.add_scalar("std_cos_c", metrics.std_cos_c[-1], epoch_idx + 1)
    writer.add_scalar("max_equi_angle", metrics.max_equi_angle[-1],
                      epoch_idx + 1)
    writer.flush()

    model_path = f"runs/{timestamp}/model_{epoch_idx + 1}"
    torch.save(model.state_dict(), model_path)

    if utils.DEBUG and epoch_idx == 2:
      break


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)
  main()
