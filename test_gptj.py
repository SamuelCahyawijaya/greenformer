import os
import shutil
import random
import numpy as np
import pandas as pd
import torch
import argparse
import json
import logging
import hashlib
import scipy
import pickle

logging.disable(logging.CRITICAL)
logging.disable(logging.WARNING)

from tqdm import tqdm
import time

import jax
from jax.experimental import maps
import numpy as np
import optax
import transformers

from mesh_transformer.checkpoint import read_ckpt
from mesh_transformer.sampling import nucleaus_sample
from mesh_transformer.transformer_shard import CausalTransformer

if __name__ == "__main__":
    ### LOAD GPT-J jax shit
    params = {
    "layers": 28,
    "d_model": 4096,
    "n_heads": 16,
    "n_vocab": 50400,
    "norm": "layernorm",
    "pe": "rotary",
    "pe_rotary_dims": 64,
    "early_cast": True,
    "seq": 2048,
    "cores_per_replica": 1,  # only running on one GPU
    "per_replica_batch": 1,
    }

    per_replica_batch = params["per_replica_batch"]
    cores_per_replica = params["cores_per_replica"]
    seq = params["seq"]

    params["sampler"] = nucleaus_sample

    # here we "remove" the optimizer parameters from the model (as we don't need them for inference)
    params["optimizer"] = optax.scale(0)

    devices = np.array([jax.devices()[0]]).reshape((1, 1))
    maps.thread_resources.env = maps.ResourceEnv(maps.Mesh(devices, ('dp', 'mp')))

    tokenizer = transformers.GPT2TokenizerFast.from_pretrained('gpt2')

    model = CausalTransformer(params)

    start = time.time()

    # here we load a checkpoint which was written with 8 shards into 1 shard
    model.state = read_ckpt(model.state, "step_383500/", 8, shards_out=cores_per_replica)

    # move the state to CPU/system memory so it's not duplicated by xmap
    model.state = jax.device_put(model.state, jax.devices("cpu")[0])
