import os
import torch
import pyiqa

curdir = os.path.abspath(os.path.join(__file__, os.pardir))

class MetricModel(torch.nn.Module):
    def __init__(self, device, model_path=f'{curdir}/../DBCNN_KonIQ10k-254e8241.pth'):
        super().__init__()
        self.model = pyiqa.create_metric(
            'dbcnn',
            as_loss=True,
            pretrained_model_path=model_path,
            device=device
        )
        self.lower_better = self.model.lower_better
        self.full_reference = False

    def forward(self, image, inference=False):
        return self.model(image)
