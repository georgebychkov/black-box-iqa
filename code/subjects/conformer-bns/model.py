import os
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))
import torch
import torch.nn as nn
from torchvision import transforms

import configs.PIPAL.IQA_Conformer as ConformerModel
import functions
sys.path.remove(str(Path(__file__).parent))

class MetricModel(torch.nn.Module):
    
    mos_mean=1448.9595
    mos_std=121.5351
    
    def __init__(self, device, model_path, backbone_path=None):
        super().__init__()
        self.device = device
        
        model = ConformerModel.model
        model.load(model_path)
        model.to(device)
        model.eval()

        self.model = model
        self.lower_better = False
    
    def forward(self, ref, dist, inference=False):
        # transforms.Compose doesn't accept torch tensors
        out = self.model(
            [
                ((transforms.Resize([192, 192])(ref)) - 0.5) / 0.5,
                ((transforms.Resize([192, 192])(dist)) - 0.5) / 0.5,
            ]
        ) * self.mos_std + self.mos_mean
        if inference:
            return out.detach().cpu().numpy()[0][0].item()
        else:
            return out

