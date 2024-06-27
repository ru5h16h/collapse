import argparse
import collections
import logging
import os

import matplotlib.pyplot as plt
import numpy as np

import utils


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--exp",
      type=str,
      required=True,
      help="Experiment ID",
  )
  return parser.parse_args()


def get_metrics_dir(exp, subset):
  return os.path.join("runs", exp, f"metrics_{subset}")


def get_fig_path(exp, metric):
  fig_path = os.path.join("runs", exp, "plots", f"{metric}.png")
  os.makedirs(os.path.dirname(fig_path), exist_ok=True)
  return fig_path


def parse_metrics(m_dir):
  metrics = collections.OrderedDict()
  for file_name in sorted(os.listdir(m_dir),
                          key=lambda fn: int(os.path.splitext(fn)[0])):
    idx = int(os.path.splitext(file_name)[0])
    file_path = os.path.join(m_dir, file_name)
    metrics[idx] = utils.load_pickle(file_path)
  return metrics


def plot_suboptimal_gap(fig_path, train_metrics, test_metrics):
  sub_mean = []
  sub_std = []
  epochs = list(train_metrics.keys())
  for train_d, test_d in zip(train_metrics.values(), test_metrics.values()):
    sub_mean.append((train_d["sub_mean"], test_d["sub_mean"]))
    sub_std.append((train_d["sub_std"], test_d["sub_std"]))
  sub_mean = np.array(sub_mean).T
  sub_std = np.array(sub_std).T

  _, axes = plt.subplots(1, 2, figsize=(12, 4))
  for idx, subset in enumerate(["train", "test"]):
    axes[idx].plot(epochs, sub_mean[idx], color='#1B2ACC')
    axes[idx].fill_between(
        epochs,
        sub_mean[idx] - sub_std[idx] / 2,
        sub_mean[idx] + sub_std[idx] / 2,
        alpha=0.2,
        edgecolor='#1B2ACC',
        facecolor='#089FFF',
        linewidth=1,
        antialiased=True,
    )
    axes[idx].set_xlabel('Epochs')
    axes[idx].set_ylabel('Sub Optimality')
    axes[idx].set_title(f"{subset.capitalize()} set")

  plt.savefig(fig_path, dpi=300)
  plt.close()


def main():
  args = parse_args()

  train_metrics_dir = get_metrics_dir(args.exp, "train")
  test_metrics_dir = get_metrics_dir(args.exp, "test")

  train_metrics = parse_metrics(train_metrics_dir)
  test_metrics = parse_metrics(test_metrics_dir)

  plot_suboptimal_gap(
      fig_path=get_fig_path(args.exp, "sub"),
      train_metrics=train_metrics,
      test_metrics=test_metrics,
  )


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)
  main()
