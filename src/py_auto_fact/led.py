import torch.nn as nn

class LED(nn.Module):
    def __init__(self, in_features, out_features, r, bias=True):
        super().__init__()
        self.led_unit = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=r, bias=False), 
            nn.Linear(in_features=r, out_features=out_features, bias=bias)
        )

    def forward(self, inputs):
        return self.led_unit(inputs)