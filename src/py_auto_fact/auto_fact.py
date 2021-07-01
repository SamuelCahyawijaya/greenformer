import copy
import torch.nn as nn
from .lr_module import LED, CED

# Input Definition
## module - nn module to be factorized
## deepcopy - deepcopy module before factorization, return new factorized copy of the model
## ignore_lower_equal_dim - skip factorization if input feature is lower or equal to rank
## fact_led_unit - flag for skipping factorization on LED and CED unit
def auto_fact(module, rank, deepcopy=False, ignore_lower_equal_dim=True, fact_led_unit=False, solver='random'):
    if deepcopy:
        module = copy.deepcopy(module)
        
    for key, child in module._modules.items():
        if not fact_led_unit and (type(child) in [LED, CED]):
            continue
            
        if type(child) == nn.Linear:
            if ignore_lower_equal_dim and (child.in_features <= rank or child.out_features <= rank):
                # Ignore if input/output features is smaller than rank to prevent factorization on low dimensional input/output vector
                continue
                
            # Replace with LED unit
            led_module = LED(child.in_features, child.out_features, r=rank, bias=child.bias is not None)
            module._modules[key] = led_module
            
            # Initialize matrix
            if solver == 'svd':
                # led_module.led_unit[0] # Initialize U
                # led_module.led_unit[1] # Initialize V
                pass
            elif solver == 'nmf':
                # led_module.led_unit[0] # Initialize U
                # led_module.led_unit[1] # Initialize V
                pass
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
                # led_module.led_unit[0] # Initialize U
                # led_module.led_unit[1] # Initialize V
                pass
            elif solver == 'nmf':
                # led_module.led_unit[0] # Initialize U
                # led_module.led_unit[1] # Initialize V
                pass
        else:
            # Perform recursive tracing
            child = auto_fact(child, rank, fact_led_unit=fact_led_unit)
    return module
