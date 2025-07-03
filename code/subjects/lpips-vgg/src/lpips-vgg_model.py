import os
import torch
import pyiqa

curdir = os.path.abspath(os.path.join(__file__, os.pardir))

class MetricModel(torch.nn.Module):
    def __init__(self, device, model_path=f'{curdir}/../LPIPS_v0.1_vgg-a78928a0.pth'):
        super().__init__()
        self.model = pyiqa.create_metric(
            'lpips',
            as_loss=True,
            pretrained_model_path=model_path,
            net='vgg',
            device=device
        )
        self.lower_better = self.model.lower_better
        self.full_reference = True

    def forward(self, ref, dist):
        return self.model(ref, dist)
