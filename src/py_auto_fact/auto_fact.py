import copy
import torch
import torch.nn as nn
import matrix_fact
from .lr_module import LED, CED

r"""
Input:
    weight - weight of the original nn.module to be factorized
    rank - the rank to be applied for low-rank factorization
Output:
    low-rank factorization weight matrix U and V
"""
def linear_snmf(weight, rank, num_iter=10):
    orig_device = weight.device
    mdl = matrix_fact.TorchSNMF(weight, rank)
    mdl.factorize(num_iter)
    return mdl.W, mdl.H

r"""
Input:
    weight - weight of the original nn.module to be factorized
    rank - the rank to be applied for low-rank factorization
Output:
    low-rank factorization weight matrix U and V
"""
def linear_nmf(weight, rank, num_iter=10):
    orig_device = weight.device
    data = weight.cpu().detach().numpy()
    mdl = matrix_fact.NMF(data, rank)
    mdl.factorize(num_iter)
    return torch.FloatTensor(mdl.W, device=orig_device), torch.FloatTensor(mdl.H, device=orig_device)

r"""
Input:
    weight - weight of the original nn.module to be factorized
    rank - the rank to be applied for low-rank factorization
Output:
   low-rank factorization weight matrix (U.S) and V
"""
def linear_svd(weight, rank, num_iter=10):
    u,s,v = torch.svd_lowrank(weight, q=rank, niter=num_iter)
    return (u*s), v.T

r"""
"""
def factorize_module(module, rank, ignore_lower_equal_dim, fact_led_unit, solver, num_iter):
    if type(module) == nn.Linear:
        if ignore_lower_equal_dim and (module.in_features <= rank or module.out_features <= rank):
            # Ignore if input/output features is smaller than rank to prevent factorization on low dimensional input/output vector
            return module

        # Create LED unit
        led_module = LED(module.in_features, module.out_features, r=rank, bias=module.bias is not None, device=module.weight.device)

        # Initialize matrix
        if solver == 'svd':
            U, V = linear_svd(module.weight.T, rank)
            led_module.led_unit[0].weight.data = U.T # Initialize U
            led_module.led_unit[1].weight.data = V.T # Initialize V
            if module.bias is not None:
                led_module.led_unit[1].bias = module.bias
        elif solver == 'nmf':
            U, V = linear_nmf(module.weight.T, rank)
            led_module.led_unit[0].weight.data = U.T # Initialize U
            led_module.led_unit[1].weight.data = V.T # Initialize V
            if module.bias is not None:
                led_module.led_unit[1].bias = module.bias
        elif solver == 'snmf':
            U, V = linear_snmf(module.weight.T, rank)
            led_module.led_unit[0].weight.data = U.T # Initialize U
            led_module.led_unit[1].weight.data = V.T # Initialize V
            if module.bias is not None:
                led_module.led_unit[1].bias = module.bias

        # Return module
        return led_module

    elif type(module) in [nn.Conv1d, nn.Conv2d, nn.Conv3d]:
        if ignore_lower_equal_dim and (module.in_channels <= rank or module.out_channels <= rank):
            # Ignore if input/output features is smaller than rank to prevent factorization on low dimensional input/output vector
            return module

        # Replace with CED unit
        ced_module = CED(module.in_channels, module.out_channels, r=rank, kernel_size=module.kernel_size, stride=module.stride, 
                            padding=module.padding, dilation=module.dilation, padding_mode=module.padding_mode, bias=module.bias is not None)

        # Initialize matrix
        if solver == 'svd':
            weight = module.weight.view(module.out_channels, -1)
            u,v = linear_svd(weight.T, rank)
            ced_module.ced_unit[0].weight.data = u.T.view_as(ced_module.ced_unit[0].weight) # Initialize U
            ced_module.ced_unit[1].weight.data = v.T.view_as(ced_module.ced_unit[1].weight) # Initialize V
            if module.bias is not None:
                ced_module.ced_unit[1].bias.data = module.bias.data
        elif solver == 'nmf':
            weight = module.weight.view(module.out_channels, -1)
            u,v = linear_nmf(weight.T, rank)
            ced_module.ced_unit[0].weight.data = u.T.view_as(ced_module.ced_unit[0].weight) # Initialize U
            ced_module.ced_unit[1].weight.data = v.T.view_as(ced_module.ced_unit[1].weight) # Initialize V
            if module.bias is not None:
                ced_module.ced_unit[1].bias.data = module.bias.data
        elif solver == 'snmf':
            weight = module.weight.view(module.out_channels, -1)
            u,v = linear_snmf(weight.T, rank)   
            ced_module.ced_unit[0].weight.data = u.T.view_as(ced_module.ced_unit[0].weight) # Initialize U
            ced_module.ced_unit[1].weight.data = v.T.view_as(ced_module.ced_unit[1].weight) # Initialize V
            if module.bias is not None:
                ced_module.ced_unit[1].bias.data = module.bias.data

        # Return module
        return ced_module
    
r"""
Input:
    module - nn.module to be factorized
    rank - the rank to be applied for low-rank factorization
    deepcopy - deepcopy module before factorization, return new factorized copy of the model
    ignore_lower_equal_dim - skip factorization if input feature is lower or equal to rank
    fact_led_unit - flag for skipping factorization on LED and CED unit
    solver - solver for network initialization ('random', 'svd', 'snmf')
    num_iter - number of iteration for  'svd' and 'snmf' solvers
    
Output:
    low-rank version of the given module (will create a model copy if `deep_copy=True`)
"""
def auto_fact(module, rank, deepcopy=False, ignore_lower_equal_dim=True, fact_led_unit=False, solver='random', num_iter=10):
    if deepcopy:
        module = copy.deepcopy(module)
        
    # If the top module is Linear or Conv, return the factorized module directly
    if type(module) in [nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d]:
        return factorize_module(module, rank, ignore_lower_equal_dim, fact_led_unit, solver, num_iter)
    
    for key, child in module._modules.items():
        if not fact_led_unit and (type(child) in [LED, CED]):
            continue
            
        if type(child) in [nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d]:
            # Replace module
            module._modules[key] = factorize_module(child, rank, ignore_lower_equal_dim, fact_led_unit, solver, num_iter)
        else:
            # Perform recursive tracing
            child = auto_fact(child, rank, fact_led_unit=fact_led_unit, solver=solver, num_iter=num_iter)
    return module
