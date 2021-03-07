import torch
import torch.nn as nn

class HLNet(nn.Module):
    def __init__(self, hlnet_base):
        super(HLNet, self).__init__()
        self.hlnet_base = hlnet_base
        self.top = nn.Linear(64, 6)
    def forward(self, HL):
        HL = self.hlnet_base(HL)
        out = self.top(HL)
        return out