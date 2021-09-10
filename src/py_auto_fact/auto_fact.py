import copy
import torch
import torch.nn as nn
import matrix_fact
from .lr_module import LED, CED
import warnings
from transformers.modeling_utils import Conv1D as HFConv1D

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
Input:
    module - nn.module to be factorized
    rank - the rank to be applied for low-rank factorization
    fact_led_unit - flag for skipping factorization on LED and CED unit
    solver - solver for network initialization ('random', 'svd', 'snmf')
    num_iter - number of iteration for  'svd' and 'snmf' solvers
    
Output:
    low-rank version of the given module
"""
def factorize_module(module, rank, fact_led_unit, solver, num_iter):
    if type(module) == nn.Linear:
        limit_rank = int((module.in_features * module.out_features) / (module.in_features + module.out_features))
        # Define rank from the given rank percentage
        if rank < 1:
            rank = int(limit_rank * rank)
            if rank == 0:
                return module
        rank = int(rank)
                    
        if (limit_rank <= rank):
            warnings.warn(f'skipping linear with in: {module.in_features}, out: {module.out_features}, rank: {rank}')
            # Ignore if input/output features is smaller than rank to prevent factorization on low dimensional input/output vector
            return module
        
        # Extract module weight
        weight = module.weight
                
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
    elif type(module) == HFConv1D:
        in_features, out_features = module.weight.shape
        limit_rank = int((in_features * out_features) / (in_features + out_features))
        # Define rank from the given rank percentage
        if rank < 1:
            rank = int(limit_rank * rank)
            if rank == 0:
                return module
        rank = int(rank)
                    
        if (limit_rank <= rank):
            warnings.warn(f'skipping linear with in: {in_features}, out: {out_features}, rank: {rank}')
            # Ignore if input/output features is smaller than rank to prevent factorization on low dimensional input/output vector
            return module
        
        # Extract module weight
        weight = module.weight.T
                
        # Create LED unit
        led_module = LED(in_features, out_features, r=rank, bias=module.bias is not None, device=module.weight.device)

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
            
        if (limit_rank <= rank):
            warnings.warn(f'skipping convolution with in: {module.in_channels}, out: {module.out_channels // module.groups}, rank: {rank}')
            # Ignore if input/output features is smaller than rank to prevent factorization on low dimensional input/output vector
            return module

        # Extract layer weight
        weight = module.weight.view(module.out_channels, -1)
        
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
    module - the module (nn.Module) to be factorized (required)
    rank - the rank to be applied for low-rank factorization (required)
    deepcopy - deepcopy module before factorization, return new factorized copy of the model (default: False)
    solver - solver for network initialization ('random', 'svd', 'snmf') (default: 'random')
    num_iter - number of iteration for  'svd' and 'snmf' solvers (default: 10)
    submodules - submodules of model of which the factorization will be applied (default: None)
    fact_led_unit - flag for skipping factorization on LED and CED unit (default: False)
    
Output:
    low-rank version of the given module (will create a model copy if `deep_copy=True`)
"""
def auto_fact(module, rank, solver='random', num_iter=10, submodules=None, deepcopy=False, fact_led_unit=False):
    if deepcopy:
        copy_module = copy.deepcopy(module)
    else:
        copy_module = module
    
    def auto_fact_recursive(module, reference_module, rank, solver, num_iter, submodules, fact_led_unit, factorize_child):
        # If the top module is Linear or Conv, return the factorized module directly
        if type(reference_module) in [nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, HFConv1D]:
            return factorize_module(module, rank, fact_led_unit, solver, num_iter)

        for key, reference_key in zip(module._modules, reference_module._modules):
            # Skip LED or CED units if `fact_led_unit` is True
            if not fact_led_unit and type(reference_module._modules[reference_key]) in [LED, CED]:
                continue

            if type(reference_module._modules[reference_key]) in [nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, HFConv1D] \
                    and (factorize_child or reference_module._modules[reference_key] in ([] if submodules is None else submodules)):
                # Factorize Linear to LED and Convolution to CED
                module._modules[key] = factorize_module(module._modules[key], rank, fact_led_unit, solver, num_iter)
            else:
                # Perform recursive tracing
                if(len(reference_module._modules[reference_key]._modules.items()) > 0):
                    if submodules is None or reference_module._modules[reference_key] in submodules:
                        module._modules[key] = auto_fact_recursive(module._modules[key], reference_module._modules[reference_key], rank, 
                                                            solver, num_iter, submodules, fact_led_unit=fact_led_unit, factorize_child=True)
                    else:
                        module._modules[key] = auto_fact_recursive(module._modules[key], reference_module._modules[reference_key], rank,
                                                    solver, num_iter, submodules, fact_led_unit=fact_led_unit, factorize_child=factorize_child)
        return module

    # Perform recursive factorization
    return auto_fact_recursive(copy_module, module, rank, solver, num_iter, submodules, fact_led_unit, False)