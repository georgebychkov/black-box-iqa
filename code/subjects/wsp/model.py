from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))
import torch
from torchvision import transforms

from iqanet import IQANet
sys.path.remove(str(Path(__file__).parent))


class MetricModel(torch.nn.Module):
    def __init__(self, device, model_path):
        super().__init__()
        self.device = device
        
        model = IQANet()
        
        model.load_state_dict(torch.load(model_path)['state_dict'])
        model.eval().to(device)
        
        self.model = model
        self.lower_better = False
    
    def forward(self, image, inference=False):
        out = self.model(
            image
        )
        out = out[:, 0]*1 + out[:, 1]*2 + out[:, 2]*3 + out[:, 3]*4 + out[:, 4]*5
        if inference:
            return out.detach().cpu().numpy()[0][0].item()
        else:
            return out
        
        