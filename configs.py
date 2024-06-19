import datetime
from typing import Any


class Configs(dict):

  def __init__(self, default_configs, args):
    default_configs["args"] = vars(args)
    self.update(default_configs)

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
