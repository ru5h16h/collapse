import json
import os
from typing import Any


class Configs(dict):

  def __init__(self, default_configs, args):
    default_configs["args"] = vars(args)
    self.update(default_configs)
    self.save_configs()

  def __getitem__(self, __key: Any) -> Any:
    if isinstance(__key, str):
      return super().__getitem__(__key)
    elif isinstance(__key, tuple):
      sub_dict = None
      for key in __key:
        if sub_dict is None:
          sub_dict = super().__getitem__(key)
        else:
          sub_dict = sub_dict[key]
      return sub_dict

  def save_configs(self):
    root_path = self["path", "root"].format(experiment=self["experiment"])
    cfg_path = os.path.join(root_path, "configs.json")
    os.makedirs(root_path, exist_ok=True)
    with open(cfg_path, "w") as fp:
      json.dump(self, fp)
