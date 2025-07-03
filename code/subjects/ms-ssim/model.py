from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))
import torch
from torchvision import transforms
from piqa import MS_SSIM
sys.path.remove(str(Path(__file__).parent))


    
class MetricModel(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        
        self.model = MS_SSIM(reduction='none').to(self.device)
        self.lower_better = False
        

    
    def forward(self, ref, dist):
        return self.model(ref.to(self.device), dist.to(self.device))
        
        