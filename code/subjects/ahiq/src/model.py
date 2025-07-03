import os
import torch
import pyiqa

curdir = os.path.abspath(os.path.join(__file__, os.pardir))

class MetricModel(torch.nn.Module):
    def __init__(self, device, 
            pretrained_model_path=f'{curdir}/../AHIQ_vit_p8_epoch33-da3ea303.pth',
            checkpoint_resnet=f'{curdir}/../resnet50_a1_0-14fe96d1.pth',
            checkpoint_vit=f'{curdir}/../B_8-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz'
            ):
        super().__init__()
        self.model = pyiqa.create_metric(
            'ahiq',
            as_loss=True,
            pretrained_model_path=pretrained_model_path,
            checkpoint_resnet=checkpoint_resnet,
            checkpoint_vit=checkpoint_vit,
            device=device
        )
        self.lower_better = self.model.lower_better

    def forward(self, ref, dist):
        return self.model(dist, ref)
