from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))
import torch
from torchvision import transforms
from nets.swin_multilevel2 import swin_FR_NR_modified
sys.path.remove(str(Path(__file__).parent))


class MetricModel(torch.nn.Module):
    def __init__(self, device, model_path):
        super().__init__()
        self.device = device
        
        model = swin_FR_NR_modified(mode='4')
        model_weight = model.state_dict()
        pretrained_weights = torch.load(model_path, map_location=device)['model']
        new_dict = {}
        for k in pretrained_weights.keys():
            if k.replace('module.', '') in model_weight:
                new_dict[k.replace('module.', '')] = pretrained_weights[k]
        model_weight.update(new_dict)
        model.load_state_dict(model_weight)
        model.eval().to(device)
        
        self.model = model
        self.lower_better = True
    
    def forward(self, ref, dist):
        size = 224
        stride = 224
        ref_patches = ref.unfold(2, size, stride).unfold(3, size, stride).reshape(-1, 3, size, size)
        dist_patches = dist.unfold(2, size, stride).unfold(3, size, stride).reshape(-1, 3, size, size)
        return self.model(ref_patches, dist_patches).mean()
        
        