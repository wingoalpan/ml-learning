import os, sys
import torch
import json as js
from safetensors.torch import (
    save_file,
)

from typing import Dict, Union

import transformers.modeling_utils as modeling_utils

from wingoal_utils.common import log

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SAFE_WEIGHTS_NAME = 'model.safetensors'
SAFE_WEIGHTS_INDEX_NAME = 'model.safetensors.index.json'


def save_safetensors(state_dict: Dict[str, torch.Tensor], save_directory: Union[str, os.PathLike],
                     max_shard_size: Union[int, str] = "45MB"):
    if os.path.isfile(save_directory):
        log("Provided path ({save_directory}) should be a directory, not a file")
        return

    os.makedirs(save_directory, exist_ok=True)
    shards, index = modeling_utils.shard_checkpoint(state_dict,
                                                    max_shard_size=max_shard_size,
                                                    weights_name=SAFE_WEIGHTS_NAME)
    # Save the model
    for shard_file, shard in shards.items():
        save_file(shard, os.path.join(save_directory, shard_file), metadata={"format": "pt"})

    if index is not None:
        save_index_file = SAFE_WEIGHTS_INDEX_NAME
        save_index_file = os.path.join(save_directory, save_index_file)
        # Save the index as well
        with open(save_index_file, "w", encoding="utf-8") as f:
            content = js.dumps(index, indent=2, sort_keys=True) + "\n"
            f.write(content)


def save_as_safetensors(state_dict_file, save_directory='models',
                        max_shard_size: Union[int, str] = "45MB"):
    state_dict = torch.load(state_dict_file, map_location=device)
    save_safetensors(state_dict, save_directory, max_shard_size=max_shard_size)


def save_pretrained(model, save_directory: Union[str, os.PathLike],
                    max_shard_size: Union[int, str] = "45MB"):
    state_dict = model.state_dict()
    save_safetensors(state_dict, save_directory, max_shard_size=max_shard_size)
