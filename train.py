#!/usr/bin/env python3

import logging
import os

import torch
from torch import optim
import torch.nn as nn
from torch.utils import tensorboard
from torch.utils import data

import utils
import evaluate

EPOCHS = 25


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


def main():
  timestamp = utils.get_current_ts()

  model = utils.load_model()

  train_loader, test_loader = utils.load_data()

  loss_fn = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

  os.makedirs(f"runs/{timestamp}")
  writer = tensorboard.writer.SummaryWriter(f"runs/{timestamp}/writer")

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

    train_loss, train_acc = evaluate.evaluate(model, train_loader, loss_fn)
    test_loss, test_acc = evaluate.evaluate(model, test_loader, loss_fn)

    logging.info(f"Epoch {epoch_idx + 1}. "
                 f"Loss: Train {train_loss:0.6f}. Val {test_loss:0.6f} "
                 f"Accuracy: Train {train_acc:0.6f}. Val {test_acc:0.6f} ")

    writer.add_scalars(
        main_tag="Training vs. Testing loss",
        tag_scalar_dict={
            "Training": train_loss,
            "Testing": test_loss
        },
        global_step=epoch_idx + 1,
    )
    writer.flush()

    model_path = f"runs/mnist/{timestamp}/models_{epoch_idx + 1}"
    torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)
  main()
