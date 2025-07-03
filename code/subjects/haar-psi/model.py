from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))
import torch
from torchvision import transforms
from piqa import HaarPSI
sys.path.remove(str(Path(__file__).parent))


    
class MetricModel(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        
        self.model = HaarPSI(reduction='none').to(self.device)
        self.lower_better = False
        

    
    def forward(self, ref, dist):
        return self.model(ref, dist)
        
        
