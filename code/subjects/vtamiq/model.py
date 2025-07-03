import os
import torch
from torchvision import transforms

from run_main import get_model
from data.sampling.patch_sampling import get_iqa_patches, PatchSampler

curdir = os.path.abspath(os.path.join(__file__, os.pardir))

class MetricModel(torch.nn.Module):
    def __init__(self, device, 
            pretrained_model_path=f'{curdir}/../latest.pth',
            ):
        super().__init__()
        
        self.device = device
        self.model = get_model(device, True, pretrained_model_path).eval()
        self.sampler = PatchSampler()
        self.lower_better = True

    def forward(self, ref, dist):
        # ref: torch tensor (1, c, h, w)
        
        ref = ref[0]
        dist = dist[0]
        patches = get_iqa_patches(
            [transforms.ToPILImage()(ref), transforms.ToPILImage()(dist)], 
            [ref,dist], 256, (16,16), self.sampler, 5)
        
        score = self.model(
            [patches[0].unsqueeze(0).to(self.device), patches[1].unsqueeze(0).to(self.device)], 
            patches_scales=patches[3].to(self.device), patches_pos=patches[2].to(self.device)).mean()
        
        return score
