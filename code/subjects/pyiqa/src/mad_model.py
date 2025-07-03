import os
import torch
import pyiqa

class MetricModel(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.model = pyiqa.create_metric(
            'mad',
            as_loss=True,
            device=device
        )
        self.lower_better = self.model.lower_better
        self.full_reference = True

    def forward(self, ref, dist):
        return self.model(ref, dist)
