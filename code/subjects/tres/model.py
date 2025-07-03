from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))
import torch
from torchvision import transforms
import argparse


from tres_models import Net
sys.path.remove(str(Path(__file__).parent))

class MetricModel(torch.nn.Module):
    def __init__(self, device, model_path):
        super().__init__()
        self.device = device
        
        config = argparse.Namespace()
        config.network = 'resnet50'
        config.nheadt = 16
        config.num_encoder_layerst = 2
        config.dim_feedforwardt = 64
        
        sys.path.append(str(Path(__file__).parent))
        model = Net(config, device).to(device)
        sys.path.remove(str(Path(__file__).parent))
        
        model.load_state_dict(torch.load(model_path))
        model.eval().to(device)
        
        self.model = model
        self.lower_better = False
    
    def forward(self, image, inference=False):
        
        out, _ = self.model(
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(transforms.Resize([224, 224])(image))
        )
        
        if inference:
            return out.detach().cpu().numpy()[0][0].item()
        else:
            return out
        
        