import copy
import torch.nn as nn
from .led import LED

def auto_fact(module, rank, deepcopy=False):
    if deepcopy:
        module = copy.deepcopy(module)
        
    for key, child in module._modules.items():
        if type(child) == LED:
            continue
        if type(child) == nn.Linear:
            module._modules[key] = LED(child.in_features, child.out_features, r=rank, bias=child.bias is not None)
        else:
            # Recursive call
            child = auto_fact(child, rank)
    return module
