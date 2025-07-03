from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))
import torch
from torchvision import transforms
import numpy as np
from Network import EONSS
sys.path.remove(str(Path(__file__).parent))




    
class MetricModel(torch.nn.Module):
    def __init__(self, device, model_path):
        super().__init__()
        self.device = device

        model = EONSS()
        model.load_state_dict(torch.load(model_path, map_location=device)['state_dict'], strict=False)
        model.eval().to(device)

        self.model = model
        self.lower_better = False
        
    def get_patches(self, image, output_size, stride):
        h, w = image.shape[-2:]
        new_h, new_w = output_size, output_size
        stride_h, stride_w = stride, stride
     
        h_start = np.arange(0, h - new_h, stride_h)
        w_start = np.arange(0, w - new_w, stride_w)

        patches = [transforms.functional.crop(image, hv_s, wv_s, new_h, new_w) for hv_s in h_start for wv_s in w_start]
        patches = torch.concatenate(patches, dim=0)
        return patches
    
    def forward(self, dist, inference=False):
        score = self.model(self.get_patches(dist, 235, 128).squeeze(1)).mean()
        return score
        
        
