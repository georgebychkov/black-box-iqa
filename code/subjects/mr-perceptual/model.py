from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))
import torch
from torchvision import transforms
from mrpl import mrpl
sys.path.remove(str(Path(__file__).parent))


    
class MetricModel(torch.nn.Module):
    def __init__(self, device, model_path):
        super().__init__()
        self.device = device
        self.lower_better = True
        self.model = mrpl.MRPL(net='alex', spatial=False, mrpl=True, eval_mode=False)
        self.model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        self.model.eval().to(device)


    
    def forward(self, ref, dist):
        if ref.shape[0] > 1:
          return self.model(
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(ref[0].unsqueeze(0)),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(dist[0].unsqueeze(0))
          )
        score = self.model(
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(ref),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(dist)
        )
        return score
        
        
