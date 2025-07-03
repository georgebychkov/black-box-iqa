import os
import torch
from stlpips_pytorch import  stlpips

curdir = os.path.abspath(os.path.join(__file__, os.pardir))

class MetricModel(torch.nn.Module):
    def __init__(self, device, 
            pretrained_model_path=f'{curdir}/../alex_shift_tolerant.pth'
            ):
        super().__init__()
        self.model = stlpips.LPIPS(net='alex', variant="shift_tolerant", pretrained=True,  model_path=pretrained_model_path).to(device)
        self.lower_better = True
        self.full_reference = True

    def forward(self, ref, dist):
        return self.model(dist, ref)
