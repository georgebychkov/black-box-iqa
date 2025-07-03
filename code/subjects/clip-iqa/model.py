from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))
import os
import torch
from torchvision import transforms

from mmedit.apis import init_model
sys.path.remove(str(Path(__file__).parent))
    
class MetricModel(torch.nn.Module):
    def __init__(self, device, model_path):
        super().__init__()
        self.device = device
        
        config = r'configs/clipiqa/clipiqa_attribute_test.py'
        model = init_model(os.path.join(Path(__file__).parent, config), model_path, device=device)
        self.model = model
        self.lower_better = False
        self.transform = transforms.Compose([
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    
    def forward(self, image, inference=False):

        data = {'meta': {}, 'lq': self.transform(image)}
        out = self.model(test_mode=True, **data)['output']
        if inference:
            return out.detach().cpu().numpy()[0][0].item()
        else:
            return out
        
        
