import copy
import torch
import torch.nn as nn
import matrix_fact
from .lr_module import LED, CED
import warnings

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
def factorize_module(module, rank, ignore_lower_equal_dim, fact_led_unit, solver, num_iter, eigen_threshold):
    if type(module) == nn.Linear:
        limit_rank = int((module.in_features * module.out_features) / (module.in_features + module.out_features))
        # Define rank from the given rank percentage
        if rank < 1:
            rank = int(limit_rank * rank)
            if rank == 0:
                return module
        rank = int(rank)
                    
        if ignore_lower_equal_dim and (limit_rank <= rank):
            warnings.warn(f'skipping linear with in: {module.in_features}, out: {module.out_features}, rank: {rank}')
            # Ignore if input/output features is smaller than rank to prevent factorization on low dimensional input/output vector
            return module
        
        # Extract module weight
        weight = module.weight
        
        if eigen_threshold is not None:
            # Perform eigen
            eigen_vals, _ = torch.linalg.eigh(weight.mm(weight.T))
            cum_eigen_vals = (eigen_vals / eigen_vals.sum()).sort(descending=True).values.cumsum(dim=0)
            if cum_eigen_vals[rank] < eigen_threshold:
                warnings.warn(f'cumulative eigen values < eigen_threshold ({eigen_threshold})')
                return module
        
        # Create LED unit
        led_module = LED(module.in_features, module.out_features, r=rank, bias=module.bias is not None, device=module.weight.device)

        # Initialize matrix
        if solver == 'random':
            pass
        elif solver == 'svd':
            U, V = linear_svd(weight.T, rank, num_iter=num_iter)
            led_module.led_unit[0].weight.data = U.T # Initialize U
            led_module.led_unit[1].weight.data = V.T # Initialize V
            if module.bias is not None:
                led_module.led_unit[1].bias = module.bias
        elif solver == 'nmf':
            U, V = linear_nmf(weight.T, rank, num_iter=num_iter)
            led_module.led_unit[0].weight.data = U.T # Initialize U
            led_module.led_unit[1].weight.data = V.T # Initialize V
            if module.bias is not None:
                led_module.led_unit[1].bias = module.bias
        elif solver == 'snmf':
            U, V = linear_snmf(weight.T, rank, num_iter=num_iter)
            led_module.led_unit[0].weight.data = U.T # Initialize U
            led_module.led_unit[1].weight.data = V.T # Initialize V
            if module.bias is not None:
                led_module.led_unit[1].bias = module.bias

        # Return module
        return led_module

    elif type(module) in [nn.Conv1d, nn.Conv2d, nn.Conv3d]:
        # Define rank from the given rank percentage
        limit_rank = int((module.in_channels * (module.out_channels // module.groups)) / (module.in_channels + (module.out_channels // module.groups)))
        
        if rank > 0 and rank < 1:
            rank = int(limit_rank * rank)                
            if rank == 0:
                return module
        rank = int(rank)
        
        # Handle grouped convolution
        if module.groups > 1 and rank % module.groups > 0:
            rank = (1 + (rank // module.groups)) * module.groups
            
        if ignore_lower_equal_dim and (limit_rank <= rank):
            warnings.warn(f'skipping convolution with in: {module.in_channels}, out: {module.out_channels // module.groups}, rank: {rank}')
            # Ignore if input/output features is smaller than rank to prevent factorization on low dimensional input/output vector
            return module

        # Extract layer weight
        weight = module.weight.view(module.out_channels, -1)
        
        if eigen_threshold is not None:
            # Perform eigen
            eigen_vals, _ = torch.linalg.eigh(weight.mm(weight.T))
            cum_eigen_vals = (eigen_vals / eigen_vals.sum()).sort(descending=True).values.cumsum(dim=0)
            if cum_eigen_vals[rank] < eigen_threshold:
                warnings.warn(f'cumulative eigen values < eigen_threshold ({eigen_threshold})')
                return module
        
        # Replace with CED unit
        ced_module = CED(module.in_channels, module.out_channels, r=rank, kernel_size=module.kernel_size, stride=module.stride, padding=module.padding, 
                dilation=module.dilation, padding_mode=module.padding_mode, groups=module.groups, bias=module.bias is not None, device=module.weight.device)

        # Initialize matrix
        if solver == 'random':
            pass
        elif solver == 'svd':
            u,v = linear_svd(weight.T, rank, num_iter=num_iter)
            ced_module.ced_unit[0].weight.data = u.T.view_as(ced_module.ced_unit[0].weight) # Initialize U
            ced_module.ced_unit[1].weight.data = v.T.view_as(ced_module.ced_unit[1].weight) # Initialize V
            if module.bias is not None:
                ced_module.ced_unit[1].bias.data = module.bias.data
        elif solver == 'nmf':
            u,v = linear_nmf(weight.T, rank, num_iter=num_iter)
            ced_module.ced_unit[0].weight.data = u.T.view_as(ced_module.ced_unit[0].weight) # Initialize U
            ced_module.ced_unit[1].weight.data = v.T.view_as(ced_module.ced_unit[1].weight) # Initialize V
            if module.bias is not None:
                ced_module.ced_unit[1].bias.data = module.bias.data
        elif solver == 'snmf':
            u,v = linear_snmf(weight.T, rank, num_iter=num_iter)   
            ced_module.ced_unit[0].weight.data = u.T.view_as(ced_module.ced_unit[0].weight) # Initialize U
            ced_module.ced_unit[1].weight.data = v.T.view_as(ced_module.ced_unit[1].weight) # Initialize V
            if module.bias is not None:
                ced_module.ced_unit[1].bias.data = module.bias.data
        else:
            raise Exception(f'Unknown solver `{solver}`')

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
def auto_fact(module, rank, deepcopy=False, ignore_lower_equal_dim=True, fact_led_unit=False, solver='random', num_iter=10, eigen_threshold=None):
    if deepcopy:
        module = copy.deepcopy(module)
        
    # If the top module is Linear or Conv, return the factorized module directly
    if type(module) in [nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d]:
        return factorize_module(module, rank, ignore_lower_equal_dim, fact_led_unit, solver, num_iter, eigen_threshold)
    
    for key, child in module._modules.items():
        if not fact_led_unit and (type(child) in [LED, CED]):
            continue
            
        if type(child) in [nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d]:
            # Replace module
            module._modules[key] = factorize_module(child, rank, ignore_lower_equal_dim, fact_led_unit, solver, num_iter, eigen_threshold)
        else:
            # Perform recursive tracing
            child = auto_fact(child, rank, False, ignore_lower_equal_dim, fact_led_unit, solver, num_iter, eigen_threshold)
    return module
