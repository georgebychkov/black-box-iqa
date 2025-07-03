from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))
import torch
from torchvision import transforms
from models.DistillationIQA import DistillationIQANet
from option_train_DistillationIQA import set_args, check_args
sys.path.remove(str(Path(__file__).parent))




    
class MetricModel(torch.nn.Module):
    def __init__(self, device, model_path):
        super().__init__()
        self.device = device
        config = set_args()
        config.studentNet_model_path = model_path
        self.config = check_args(config)
        model = DistillationIQANet(self_patch_num=self.config.self_patch_num, distillation_layer=self.config.distillation_layer, pretrained_backbone=False)
        model._load_state_dict(torch.load(model_path))
        model.eval().to(device)
        
        self.transform = transforms.Compose([
                    transforms.RandomCrop(size=self.config.patch_size),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                         std=(0.229, 0.224, 0.225))
                ])

        self.model = model
        self.lower_better = False
        
    def preprocess(self, img):
        patches = []
        for _ in range(self.config.self_patch_num):
            patch = self.transform(img)
            patches.append(patch)
        patches = torch.cat(patches, 0)
        return patches.unsqueeze(0)
    
    def forward(self, ref, dist):
        ref_patches, dist_patches = self.preprocess(ref.contiguous()), self.preprocess(dist.contiguous())
        _, _, score = self.model(dist_patches, ref_patches)
        return score
        
        
