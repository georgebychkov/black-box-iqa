from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))
import torch
from torchvision import transforms
from torch.autograd import Variable
from siamunet_conc import SiamUnet_conc
sys.path.remove(str(Path(__file__).parent))


class MetricModel(torch.nn.Module):
    def __init__(self, device, model_path):
        super().__init__()
        self.device = device
        
        model = SiamUnet_conc(3, 1)
        
        
        model.load_state_dict(torch.load(model_path)['model_state_dict'])
        model.eval().to(device)
        
        self.model = model
        self.lower_better = False
        self.norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.resize = transforms.Resize([288, 288])
    
    def forward(self, ref, dist):
        return self.model(self.norm(self.resize(dist)), self.norm(self.resize(ref)))
        
        