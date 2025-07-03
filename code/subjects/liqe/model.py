from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))
import torch
from torchvision import transforms
from LIQE import LIQE
sys.path.remove(str(Path(__file__).parent))


class MetricModel(torch.nn.Module):
    def __init__(self, device, model_path):
        super().__init__()
        self.device = device
        
        self.model = LIQE(model_path, device)
        
        self.lower_better = False

    
    def forward(self, dist, inference=False):
        score, _, _ = self.model(dist)
        return score
        
        