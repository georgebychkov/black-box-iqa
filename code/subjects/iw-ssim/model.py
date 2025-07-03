from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))
import torch
from torchvision import transforms
from piq import information_weighted_ssim
sys.path.remove(str(Path(__file__).parent))


    
class MetricModel(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.lower_better = False
        

    
    def forward(self, ref, dist):
        try:
            return information_weighted_ssim(ref, dist, reduction='none')
        except torch._C._LinAlgError:
            return None
            
        