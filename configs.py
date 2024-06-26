import os
from typing import Any

import utils


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
    root_path = utils.get_path(self, "root")
    cfg_path = os.path.join(root_path, "configs.json")
    os.makedirs(root_path, exist_ok=True)
    utils.write_json(cfg_path, self)
