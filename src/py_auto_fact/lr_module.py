import torch.nn as nn

class LED(nn.Module):
    def __init__(self, in_features, out_features, r, bias=True, device='cpu'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.led_unit = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=r, bias=False, device=device), 
            nn.Linear(in_features=r, out_features=out_features, bias=bias, device=device)
        )

    def forward(self, inputs):
        outputs_shape = None
        if len(inputs.shape) > 2:
            outputs_shape = list(inputs.shape[:-1]) + [self.out_features]
            inputs = inputs.view(-1,inputs.shape[-1])
            
        outputs = self.led_unit(inputs)
        if outputs_shape is not None:
            outputs = outputs.view(outputs_shape)
            
        return outputs
    
class CED(nn.Module):
    def __init__(self, in_channels, out_channels, r, kernel_size, stride=1, padding=0, 
                 dilation=1, groups=1, padding_mode='zeros', bias=True, device='cpu'):
        super().__init__()
        
        module_cls = None
        fact_ks = None
        if len(kernel_size) == 1:
            module_cls = nn.Conv1d
            fact_ks = (1,)
        elif len(kernel_size) == 2:
            module_cls = nn.Conv2d
            fact_ks = (1,1)
        elif len(kernel_size) == 3:
            module_cls = nn.Conv3d
            fact_ks = (1,1,1)
        else:
            raise ValueError(f'invalid dimension for parameter `kernel_size`. Only 1d, 2d, and 3d kernel size is supported')

        self.ced_unit = nn.Sequential(
            module_cls(in_channels=in_channels, out_channels=r, kernel_size=kernel_size, stride=stride, padding=padding, 
                       dilation=dilation, groups=groups, padding_mode=padding_mode, bias=False, device=device),
            module_cls(in_channels=r, out_channels=out_channels, kernel_size=fact_ks, bias=bias, device=device)
        )
            
    def forward(self, inputs):
        return self.ced_unit(inputs)
