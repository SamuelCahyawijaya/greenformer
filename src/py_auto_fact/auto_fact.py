import copy
import torch
import torch.nn as nn
import pymf3
from .lr_module import LED, CED

r"""
Input:
    weight - weight of the original nn.module to be factorized
    rank - the rank to be applied for low-rank factorization
Output:
    low-rank factorization weight matrix U and V
"""
def linear_snmf(weight, rank, num_iter=10):
    data = matrix.cpu().numpy()
    mdl = pymf.SNMF(data, rank)
    mdl.factorize(niter)
    return torch.FloatTensor(mdl.W, device.weight.device), torch.FloatTensor(mdl.H, device.weight.device)

r"""
Input:
    weight - weight of the original nn.module to be factorized
    rank - the rank to be applied for low-rank factorization
Output:
   low-rank factorization weight matrix U and V
"""
def linear_svd(weight, rank, num_iter=10):
    u,s,v = torch.svd_lowrank(weight, q=rank, niter=num_iter)
    return u*s, v.T
    
r"""
Input:
    module - nn.module to be factorized
    rank - the rank to be applied for low-rank factorization
    deepcopy - deepcopy module before factorization, return new factorized copy of the model
    ignore_lower_equal_dim - skip factorization if input feature is lower or equal to rank
    fact_led_unit - flag for skipping factorization on LED and CED unit
    solver - solver for network initialization ('random', 'svd', 'nmf')
    num_iter - number of iteration for  'svd' and 'snmf' solvers
    
Output:
    low-rank version of the given module (will create a model copy if `deep_copy=True`)
"""
def auto_fact(module, rank, deepcopy=False, ignore_lower_equal_dim=True, fact_led_unit=False, solver='random', num_iter=5):
    if deepcopy:
        module = copy.deepcopy(module)
        
    for key, child in module._modules.items():
        if not fact_led_unit and (type(child) in [LED, CED]):
            continue
            
        if type(child) == nn.Linear:
            if ignore_lower_equal_dim and (child.in_features <= rank or child.out_features <= rank):
                # Ignore if input/output features is smaller than rank to prevent factorization on low dimensional input/output vector
                continue
                
            # Create LED unit
            led_module = LED(child.in_features, child.out_features, r=rank, bias=child.bias is not None, device=child.weight.device)
            
            # Initialize matrix
            if solver == 'svd':
                U, V = linear_svd(child.weight, rank)
                led_module.led_unit[0].weight.data = U # Initialize U
                led_module.led_unit[1].weight.data = V # Initialize V
                led_module.led_unit[1].bias = child.bias
            elif solver == 'nmf':
                # led_module.led_unit[0] # Initialize U
                # led_module.led_unit[1] # Initialize V
                led_module.led_unit[1].bias = child.bias
            elif solver == 'snmf':
                led_module.led_unit[0], led_module.led_unit[1] = linear_snmf(module.weight, rank)

            # Replace module
            module._modules[key] = led_module
                
        elif type(child) in [nn.Conv1d, nn.Conv2d, nn.Conv3d]:
            if ignore_lower_equal_dim and (child.in_channels <= rank or child.out_channels <= rank):
                # Ignore if input/output features is smaller than rank to prevent factorization on low dimensional input/output vector
                continue
                
            # Replace with CED unit
            ced_module = CED(child.in_channels, child.out_channels, r=rank, kernel_size=child.kernel_size, stride=child.stride, 
                                        padding=child.padding, dilation=child.dilation, padding_mode=child.padding_mode, bias=child.bias is not None)
            module._modules[key] = ced_module
            
            # Initialize matrix
            if solver == 'svd':
                raise NotImplementedError
                # ced_module.led_unit[0] # Initialize U
                # ced_module.led_unit[1] # Initialize V
                ced_module.led_unit[1].bias = module.bias
            elif solver == 'nmf':
                raise NotImplementedError
                # ced_module.led_unit[0] # Initialize U
                # ced_module.led_unit[1] # Initialize V
                ced_module.led_unit[1].bias = module.bias
        else:
            # Perform recursive tracing
            child = auto_fact(child, rank, fact_led_unit=fact_led_unit, solver=solver, num_iter=num_iter)
    return module
