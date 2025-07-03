import os
import sys
curdir = os.path.abspath(os.path.join(__file__, os.pardir))
sys.path.append(curdir)

import torch
from torchvision import transforms
from BaseCNN import BaseCNN
from Transformers import AdaptiveResize
from PIL import Image


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class MetricModel(torch.nn.Module):
    def __init__(self, device, 
                resnet_path=f'{curdir}/../resnet34-b627a593.pth',
                model_path=f'{curdir}/../unique.pt'):
        super().__init__()
        self.lower_better = False
        self.full_reference = False
        
        self.test_transform = transforms.Compose([
            # AdaptiveResize(768),
            #transforms.Resize(768),
            # transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                std=(0.229, 0.224, 0.225))
        ])
        
        config = {'train': True, 'get_scores': False, 'use_cuda': True, 
                  'resume': False, 'seed': 19901116, 'backbone': 'resnet34', 
                  'fc': True, 'scnn_root': 'saved_weights/scnn.pkl', 
                  'network': 'basecnn', 'representation': 'BCNN', 'ranking': True, 
                  'fidelity': True, 'std_modeling': True, 'std_loss': True, 
                  'margin': 0.025, 'split': 1, 'trainset': './IQA_database/', 
                  'live_set': './IQA_database/databaserelease2/', 
                  'csiq_set': './IQA_database/CSIQ/', 
                  'tid2013_set': './IQA_database/TID2013/', 
                  'bid_set': './IQA_database/BID/', 
                  'clive_set': './IQA_database/ChallengeDB_release/', 
                  'koniq10k_set': './IQA_database/koniq-10k/', 
                  'kadid10k_set': './IQA_database/kadid10k/', 
                  'eval_live': True, 'eval_csiq': True, 'eval_tid2013': True, 
                  'eval_kadid10k': True, 'eval_bid': True, 'eval_clive': True, 
                  'eval_koniq10k': True, 'split_modeling': False, 
                  'ckpt_path': './checkpoint', 'ckpt': None, 
                  'train_txt': 'train.txt', 'batch_size': 128, 
                  'batch_size2': 32, 'image_size': 384, 'max_epochs': 3, 
                  'max_epochs2': 12, 'lr': 0.0001, 'decay_interval': 3, 
                  'decay_ratio': 0.1, 'epochs_per_eval': 1, 'epochs_per_save': 1
        }
        
        config = dotdict(config)
        config.backbone = 'resnet34'
        config.backbone_pretrained_path = resnet_path
        config.representation = 'BCNN'

        model = BaseCNN(config)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(model_path))
        self.model = model.eval().to(device)

    def forward(self, image, inference=False):
        # image: torch tensor (b, c, h, w)
        image = self.test_transform(image)
        score, _std = self.model(image)
        return score
