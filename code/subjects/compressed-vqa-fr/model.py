from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))
import torch
from torchvision import transforms
import UGCVQA_FR_model
sys.path.remove(str(Path(__file__).parent))


class MetricModel(torch.nn.Module):
    def __init__(self, device, model_path):
        super().__init__()
        self.device = device
        
        model = UGCVQA_FR_model.ResNet50(pretrained=False)
        state_dict = torch.load(model_path, map_location=device)
        state_dict = dict((key.split('.', 1)[1], value) for (key, value) in state_dict.items())
        model.load_state_dict(state_dict)
        model.eval().to(device)
        
        self.model = model
        self.lower_better = False
        self.transformations = transforms.Compose([transforms.Resize(520),transforms.CenterCrop(448),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])

    
    def forward(self, ref, dist):
        return self.model(self.transformations(dist).unsqueeze(dim=0), self.transformations(ref).unsqueeze(dim=0))
        
        