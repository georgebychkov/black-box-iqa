from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))
import torch
from torchvision import transforms

import models
sys.path.remove(str(Path(__file__).parent))

class MetricModel(torch.nn.Module):
    def __init__(self, device, model_path):
        super().__init__()
        self.device = device
        
        model = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7)
        
        model.load_state_dict(torch.load(model_path))
        model.eval().to(device)
        
        self.model = model
        self.lower_better = False
    
    def forward(self, image, inference=False):
        paras = self.model(
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(transforms.Resize([224, 224])(image.contiguous()))
        )
        model_target = models.TargetNet(paras).eval().to(self.device)
        out = model_target(paras['target_in_vec'])
        if inference:
            return out.detach().cpu().numpy()[0][0].item()
        else:
            return out
        
        