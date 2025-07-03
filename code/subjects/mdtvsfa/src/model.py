import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
from torchvision import models
from torchvision import transforms


class VQAModel(nn.Module):
    def __init__(self, scale={'K': 1, 'C': 1, 'L': 1, 'N': 1}, m={'K': 0, 'C': 0, 'L': 0, 'N': 0}, 
                 simple_linear_scale=False, input_size=4096, reduced_size=128, hidden_size=32):
        super(VQAModel, self).__init__()
        self.hidden_size = hidden_size
        mapping_datasets = scale.keys()

        self.dimemsion_reduction = nn.Linear(input_size, reduced_size)
        self.feature_aggregation = nn.GRU(reduced_size, hidden_size, batch_first=True)
        self.regression = nn.Linear(hidden_size, 1)
        self.bound = nn.Sigmoid()
        self.nlm = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid(), nn.Linear(1, 1))  # 4 parameters
        # self.nlm = nn.Sequential(nn.Sequential(nn.Linear(1, 1), nn.Sigmoid(), nn.Linear(1, 1, bias=False)),
        #                          nn.Linear(1, 1))  # 5 parameters
        self.lm = nn.Sequential(OrderedDict([(dataset, nn.Linear(1, 1)) for dataset in mapping_datasets]))

        torch.nn.init.constant_(self.nlm[0].weight, 2*np.sqrt(3))
        torch.nn.init.constant_(self.nlm[0].bias, -np.sqrt(3))
        torch.nn.init.constant_(self.nlm[2].weight, 1)
        torch.nn.init.constant_(self.nlm[2].bias, 0)
        for p in self.nlm[2].parameters():
            p.requires_grad = False
        for d, dataset in enumerate(mapping_datasets):
            torch.nn.init.constant_(self.lm._modules[dataset].weight, scale[dataset])
            torch.nn.init.constant_(self.lm._modules[dataset].bias, m[dataset])


        if simple_linear_scale:
            for p in self.lm.parameters():
                p.requires_grad = False

    def forward(self, x):
        x_len = x.shape[1] * torch.ones(1, 1, dtype=torch.long)
        x = self.dimemsion_reduction(x)  # dimension reduction
        x, _ = self.feature_aggregation(x, self._get_initial_state(x.size(0), x.device))
        q = self.regression(x)  # frame quality
  
        relative_score = self._sitp(q[0, :x_len[0].item()])  # video overall quality
        relative_score = torch.unsqueeze(relative_score, 0) 
        relative_score = self.bound(relative_score)
        mapped_score = self.nlm(relative_score) # 4 parameters
           

        return relative_score, mapped_score

    def _sitp(self, q, tau=12, beta=0.5):
        """subjectively-inspired temporal pooling"""
        q = torch.unsqueeze(torch.t(q), 0)
        qm = -float('inf')*torch.ones((1, 1, tau-1)).to(q.device)
        qp = 10000.0 * torch.ones((1, 1, tau - 1)).to(q.device)  #
        l = -F.max_pool1d(torch.cat((qm, -q), 2), tau, stride=1)
        m = F.avg_pool1d(torch.cat((q * torch.exp(-q), qp * torch.exp(-qp)), 2), tau, stride=1)
        n = F.avg_pool1d(torch.cat((torch.exp(-q), torch.exp(-qp)), 2), tau, stride=1)
        m = m / n
        q_hat = beta * m + (1 - beta) * l
        return torch.mean(q_hat)

    def _get_initial_state(self, batch_size, device):
        h0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        return h0
    



class CNNModel(torch.nn.Module):
    """Modified CNN models for feature extraction"""
    def __init__(self, model='ResNet-50', weights_path=None):
        super(CNNModel, self).__init__()   
        
        if weights_path:
            backbone = models.__dict__[model]()
            backbone.load_state_dict(torch.load(weights_path))
        else:
            backbone = models.__dict__[model](pretrained=True)
            
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        

    def forward(self, x):
        x = self.features(x)
        features_mean = nn.functional.adaptive_avg_pool2d(x, 1)
        features_std = global_std_pool2d(x)
        return features_mean, features_std



def global_std_pool2d(x):
    """2D global standard variation pooling"""
    return torch.std(x.view(x.size()[0], x.size()[1], -1, 1),
                     dim=2, keepdim=True)


def get_features(video_data, frame_batch_size=64, extractor=None, device='cuda'):
    """feature extraction"""
    video_length = video_data.shape[0]
    frame_start = 0
    frame_end = frame_start + frame_batch_size
    output1 = torch.Tensor().to(device)
    output2 = torch.Tensor().to(device)
    extractor.eval()
    while frame_end < video_length:
        batch = video_data[frame_start:frame_end].to(device)
        features_mean, features_std = extractor(batch)
        output1 = torch.cat((output1, features_mean), 0)
        output2 = torch.cat((output2, features_std), 0)
        frame_end += frame_batch_size
        frame_start += frame_batch_size

    last_batch = video_data[frame_start:video_length].to(device)
    features_mean, features_std = extractor(last_batch)
    output1 = torch.cat((output1, features_mean), 0)
    output2 = torch.cat((output2, features_std), 0)
    output = torch.cat((output1, output2), 1).squeeze(3).squeeze(2)

    return output



class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)
    def forward(self, x):
        return (x - self.mean.type_as(x)[None,:,None,None]) / self.std.type_as(x)[None,:,None,None]
    
    
    
class MetricModel(torch.nn.Module):
    def __init__(self, device, model_path, backbone_path=None):
        super().__init__()
        self.device = device

        model = VQAModel()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        self.backbone_path = backbone_path
        self.model = model.to(device)
        self.lower_better = False
        self.extractor = CNNModel(model='resnet50', weights_path=self.backbone_path).to(device)
    
    def forward(self, image, inference=False):
        features = get_features(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image), extractor=self.extractor, frame_batch_size=4, device=self.device)
        features = torch.unsqueeze(features, 0)
        torch.backends.cudnn.enabled = False
        _, out = self.model(features)
        torch.backends.cudnn.enabled = True
        if inference:
            return out.detach().cpu().numpy()[0][0].item()
        else:
            return out
        
    
