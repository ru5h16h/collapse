#!/usr/bin/env python3

import argparse
import functools
import logging
import os

import tqdm
import torch
from torch import optim
import torch.nn as nn
from torch.utils import tensorboard
from torch.utils import data

import configs
import utils
import evaluate

_CFG = {
    "seed": 42,
    "experiment": utils.get_current_ts(),
    "model": {
        "name": "resnet18",
    },
    "train": {
        "epochs": 350,
        "optimizer": {
            "lr": 0.0697,
            "momentum": 0.9,
            "weight_decay": 5e-4,
        },
        "lr_schedule": {
            "gamma": 0.1,
            "milestones": [0.33, 0.67],
        },
        "batch_size": 128
    },
    "evaluate": {
        "epochs": [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 20,
            22, 24, 27, 29, 32, 35, 38, 42, 45, 50, 54, 59, 65, 71, 77, 85, 92,
            101, 110, 121, 132, 144, 158, 172, 188, 206, 225, 245, 268, 293,
            320, 349
        ],
    },
    "data": {
        "img_size": 28,
        "padded_img_size": 32,
        "input_channels": 1,
        "norm_mean": 0.1307,
        "norm_std": 0.3081,
        "n_classes": 10,
    },
    "path": {
        "root": "runs/{experiment}",
        "model": "model_{epoch_idx}",
        "writer": "writer",
    }
}


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--debug",
      action="store_true",
      help="toggles debug mode",
  )
  return parser.parse_args()


def train_epoch(
    epoch_idx: int,
    model: nn.Module,
    train_loader: data.DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    debug: bool,
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

    if debug and idx == 20:
      break
  p_bar.close()


def main():
  args = parse_args()
  cfg = configs.Configs(_CFG, args)
  logging.info(f"Experiment: {cfg['experiment']}.")

  model = utils.load_model(
      model_name=cfg["model", "name"],
      in_channels=cfg["data", "input_channels"],
  )
  features = utils.Features(module=model.fc)

  train_loader, test_loader = utils.load_data(cfg)

  loss_fn = nn.CrossEntropyLoss()
  loss_fn_red = nn.CrossEntropyLoss(reduction='sum')

  optimizer = optim.SGD(params=model.parameters(), **cfg["train", "optimizer"])
  lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                **cfg["train", "lr_schedule"])

  os.makedirs(utils.get_path(cfg, "root"), exist_ok=True)
  writer = tensorboard.writer.SummaryWriter(utils.get_path(cfg, "writer"))

  metrics = utils.Metrics(tb_writer=writer)
  epochs = cfg["train", "epochs"]
  log_at_epochs = cfg["evaluate", "epochs"]
  model_path = utils.get_path(cfg, "model")

  for epoch_idx in range(epochs):
    train_epoch(
        epoch_idx=epoch_idx,
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        debug=cfg["args", "debug"],
    )
    lr_scheduler.step()

    if epoch_idx in log_at_epochs:
      evaluate_p = functools.partial(
          evaluate.evaluate,
          epoch_idx=epoch_idx,
          model=model,
          loss_fn_red=loss_fn_red,
          metrics=metrics,
          features=features,
          cfg=cfg,
      )
      evaluate_p(data_loader=train_loader, loader_type="train")
      evaluate_p(data_loader=test_loader, loader_type="test")

      torch.save(model.state_dict(), model_path.format(epoch_idx=epoch_idx))

    writer.flush()
    if cfg["args", "debug"] and epoch_idx == 2:
      break
    logging.info("_" * 79 + "\n")


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)
  main()
