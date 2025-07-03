from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))
import torch
from torchvision import transforms
from IQAmodel import *
sys.path.remove(str(Path(__file__).parent))




    
class MetricModel(torch.nn.Module):
    def __init__(self, device, model_path):
        super().__init__()
        self.device = device
        model = Model_Joint(return_feature=True, pretrained_backbone=False).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model'], strict=False)
        model.eval().to(device)
        self.k = checkpoint['k']
        self.b = checkpoint['b']


        self.model = model
        self.lower_better = False
        

    def forward(self, dist, inference=False):
        score = self.model(
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(transforms.Resize([640, 480])(dist))
        )[-1] * self.k[-1] + self.b[-1]
        return score
        
        