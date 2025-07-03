import sys
sys.path.append("src")
import torch
from torchvision import transforms
import argparse

from torch import nn


class CNNIQAnet(nn.Module):
    def __init__(self, ker_size=7, n_kers=50, n1_nodes=800, n2_nodes=800):
        super(CNNIQAnet, self).__init__()
        self.conv1  = nn.Conv2d(1, n_kers, ker_size)
        self.fc1    = nn.Linear(2 * n_kers, n1_nodes)
        self.fc2    = nn.Linear(n1_nodes, n2_nodes)
        self.fc3    = nn.Linear(n2_nodes, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        x  = x.view(-1, x.size(-3), x.size(-2), x.size(-1))  #

        h  = self.conv1(x)

        # h1 = F.adaptive_max_pool2d(h, 1)
        # h2 = -F.adaptive_max_pool2d(-h, 1)
        h1 = nn.MaxPool2d((h.size(-2), h.size(-1)))(h)
        h2 = -nn.MaxPool2d((h.size(-2), h.size(-1)))(-h)
        h  = torch.cat((h1, h2), 1)  # max-min pooling
        h  = h.squeeze(3).squeeze(2)

        h  = nn.ReLU()(self.fc1(h))
        h  = self.dropout(h)
        h  = nn.ReLU()(self.fc2(h))

        q  = self.fc3(h)
        return q



class MetricModel(torch.nn.Module):
    def __init__(self, device, model_path):
        super().__init__()
        self.device = device
        
        model = CNNIQAnet().to(device)
        
        model.load_state_dict(torch.load(model_path))
        model.eval().to(device)
        
        self.model = model
        
        
    def LocalNormalization(self, patch, P=3, Q=3, C=1):
        
        kernel = torch.ones(1, 1, P, Q).to(self.device) / (P * Q)
        patch_mean = torch.nn.functional.conv2d(transforms.Pad(1, padding_mode='symmetric')(patch), kernel, padding='valid')
        patch_sm = torch.nn.functional.conv2d(transforms.Pad(1, padding_mode='symmetric')(torch.square(patch)), kernel, padding='valid')
        patch_std = torch.sqrt(torch.maximum(patch_sm - torch.square(patch_mean), torch.zeros(1).to(self.device))) + C
        patch_ln = (patch - patch_mean) / patch_std

        return patch_ln
    
    def forward(self, image, inference=False):
        image = transforms.Grayscale()(image)
        '''
        patch_size = 32
        patches = image.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.reshape(-1, 1, patch_size, patch_size)
        patches = self.LocalNormalization(patches)
        patches = patches.reshape(-1, 1, patch_size, patch_size)
        '''
        image = self.LocalNormalization(image)
        out = self.model(
            image
            .to(self.device)
        )

        if inference:
            return out.detach().cpu().numpy()[0][0].item()
        else:
            return out
        
        