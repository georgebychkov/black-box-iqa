from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))
import torch
from torchvision import transforms
from piq import dss
sys.path.remove(str(Path(__file__).parent))


    
class MetricModel(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.lower_better = False
        

    
    def forward(self, ref, dist):
        return dss(ref, dist, reduction='none')
        
        