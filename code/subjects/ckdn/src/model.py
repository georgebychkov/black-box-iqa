import os
import torch
import pyiqa

curdir = os.path.abspath(os.path.join(__file__, os.pardir))

class MetricModel(torch.nn.Module):
    def __init__(self, device,
            pretrained_model_path=f'{curdir}/../CKDN_model_best-38b27dc6.pth',
            checkpoint_resnet=f'{curdir}/../resnet50-19c8e357.pth',
            ):
        super().__init__()
        self.model = pyiqa.create_metric(
            'ckdn',
            as_loss=True,
            pretrained_model_path=pretrained_model_path,
            checkpoint_resnet=checkpoint_resnet,
            device=device
        )
        self.lower_better = self.model.lower_better

    def forward(self, ref, dist):
        return self.model(dist, ref)
