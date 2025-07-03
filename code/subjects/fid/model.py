from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))
import torch
from torchvision import transforms
from piqa import FID
sys.path.remove(str(Path(__file__).parent))


    
class MetricModel(torch.nn.Module):
    def __init__(self, device, model_path):
        super().__init__()
        self.device = device
        
        self.model = FID(model_path).to(self.device)
        self.lower_better = True
        

    
    def forward(self, ref, dist):
        ref_features = self.model.features(ref, no_grad=False)
        dist_features = self.model.features(dist, no_grad=False)
        try:
            return self.model(ref_features, dist_features)
        except torch._C._LinAlgError:
            return None
        
        
