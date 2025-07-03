import torch
import decord
import numpy as np
import yaml
import os
from pathlib import Path
import sys
from src.dover.datasets import UnifiedFrameSampler, spatial_temporal_view_decomposition
from src.dover.models import DOVER
from src.dover.datasets import *

mean, std = (
    torch.FloatTensor([123.675, 116.28, 103.53]),
    torch.FloatTensor([58.395, 57.12, 57.375]),
)


fusion = True

def fuse_results(results: list):
    x = (results[0] - 0.1107) / 0.07355 * 0.6104 + (
        results[1] + 0.08285
    ) / 0.03774 * 0.3896
    #print(x)
    return 1 / (1 + np.exp(-x))


opt = {'name': 'DOVER',
 'num_epochs': 0,
 'l_num_epochs': 10,
 'warmup_epochs': 2.5,
 'ema': True,
 'save_model': True,
 'batch_size': 8,
 'num_workers': 6,
 'split_seed': 42,
 'wandb': {'project_name': 'DOVER'},
 'data': {'val-livevqc': {'type': 'ViewDecompositionDataset',
   'args': {'weight': 0.598,
    'phase': 'test',
    'anno_file': './examplar_data_labels/LIVE_VQC/labels.txt',
    'data_prefix': '../datasets/LIVE_VQC/',
    'sample_types': {'technical': {'fragments_h': 7,
      'fragments_w': 7,
      'fsize_h': 32,
      'fsize_w': 32,
      'aligned': 32,
      'clip_len': 32,
      'frame_interval': 2,
      'num_clips': 3},
     'aesthetic': {'size_h': 224,
      'size_w': 224,
      'clip_len': 32,
      'frame_interval': 2,
      't_frag': 32,
      'num_clips': 1}}}},
  'val-kv1k': {'type': 'ViewDecompositionDataset',
   'args': {'weight': 0.54,
    'phase': 'test',
    'anno_file': './examplar_data_labels/KoNViD/labels.txt',
    'data_prefix': '../datasets/KoNViD/',
    'sample_types': {'technical': {'fragments_h': 7,
      'fragments_w': 7,
      'fsize_h': 32,
      'fsize_w': 32,
      'aligned': 32,
      'clip_len': 32,
      'frame_interval': 2,
      'num_clips': 3},
     'aesthetic': {'size_h': 224,
      'size_w': 224,
      'clip_len': 32,
      'frame_interval': 2,
      't_frag': 32,
      'num_clips': 1}}}},
  'val-ltest': {'type': 'ViewDecompositionDataset',
   'args': {'weight': 0.603,
    'phase': 'test',
    'anno_file': './examplar_data_labels/LSVQ/labels_test.txt',
    'data_prefix': '../datasets/LSVQ/',
    'sample_types': {'technical': {'fragments_h': 7,
      'fragments_w': 7,
      'fsize_h': 32,
      'fsize_w': 32,
      'aligned': 32,
      'clip_len': 32,
      'frame_interval': 2,
      'num_clips': 3},
     'aesthetic': {'size_h': 224,
      'size_w': 224,
      'clip_len': 32,
      'frame_interval': 2,
      't_frag': 32,
      'num_clips': 1}}}},
  'val-l1080p': {'type': 'ViewDecompositionDataset',
   'args': {'weight': 0.62,
    'phase': 'test',
    'anno_file': './examplar_data_labels/LSVQ/labels_1080p.txt',
    'data_prefix': '../datasets/LSVQ/',
    'sample_types': {'technical': {'fragments_h': 7,
      'fragments_w': 7,
      'fsize_h': 32,
      'fsize_w': 32,
      'aligned': 32,
      'clip_len': 32,
      'frame_interval': 2,
      'num_clips': 3},
     'aesthetic': {'size_h': 224,
      'size_w': 224,
      'clip_len': 32,
      'frame_interval': 2,
      't_frag': 32,
      'num_clips': 1}}}},
  'val-cvd2014': {'type': 'ViewDecompositionDataset',
   'args': {'weight': 0.576,
    'phase': 'test',
    'anno_file': './examplar_data_labels/CVD2014/labels.txt',
    'data_prefix': '../datasets/CVD2014/',
    'sample_types': {'technical': {'fragments_h': 7,
      'fragments_w': 7,
      'fsize_h': 32,
      'fsize_w': 32,
      'aligned': 32,
      'clip_len': 32,
      'frame_interval': 2,
      'num_clips': 3},
     'aesthetic': {'size_h': 224,
      'size_w': 224,
      'clip_len': 32,
      'frame_interval': 2,
      't_frag': 32,
      'num_clips': 1}}}},
  'val-ytugc': {'type': 'ViewDecompositionDataset',
   'args': {'weight': 0.443,
    'phase': 'test',
    'anno_file': './examplar_data_labels/YouTubeUGC/labels.txt',
    'data_prefix': '../datasets/YouTubeUGC/',
    'sample_types': {'technical': {'fragments_h': 7,
      'fragments_w': 7,
      'fsize_h': 32,
      'fsize_w': 32,
      'aligned': 32,
      'clip_len': 32,
      'frame_interval': 2,
      'num_clips': 3},
     'aesthetic': {'size_h': 224,
      'size_w': 224,
      'clip_len': 32,
      'frame_interval': 2,
      't_frag': 32,
      'num_clips': 1}}}}},
 'model': {'type': 'DOVER',
  'args': {'backbone': {'technical': {'type': 'swin_tiny_grpb',
     'checkpoint': True,
     'pretrained': None},
    'aesthetic': {'type': 'conv_tiny'}},
   'backbone_preserve_keys': 'technical,aesthetic',
   'divide_head': True,
   'vqa_head': {'in_channels': 768, 'hidden_channels': 64}}},
 'optimizer': {'lr': 0.001, 'backbone_lr_mult': 0.1, 'wd': 0.05},
 'test_load_path': './pretrained_weights/DOVER.pth'}


class MetricModel(torch.nn.Module):
    def __init__(self, device, model_path):
        super().__init__()
        self.device = device
        self.model = DOVER(**opt["model"]["args"]).to(device)
        self.model.load_state_dict(
              torch.load(model_path, map_location=device)
        )
        self.mean = mean.to(device)
        self.std = std.to(device)

        self.dopt = opt["data"]["val-l1080p"]["args"]

        self.temporal_samplers = {}
        for stype, sopt in self.dopt["sample_types"].items():
          if "t_frag" not in sopt:
            self.temporal_samplers[stype] = UnifiedFrameSampler(sopt["clip_len"], sopt["num_clips"], sopt["frame_interval"])
          else:
            self.temporal_samplers[stype] = UnifiedFrameSampler(
            sopt["clip_len"] // sopt["t_frag"],
            sopt["t_frag"],
            sopt["frame_interval"],
            sopt["num_clips"],
          )

        self.model = self.model.to(self.device)
        self.lower_better = False


    def spatial_temporal_view_decomposition_grad(self, batch):
        batch = batch.permute(1, 0, 2, 3)


        sampled_video = {}
        sampled_video['aesthetic'] = torchvision.transforms.Resize((224, 224))(batch)
        sampled_video['technical'] = get_spatial_fragments(batch, aligned=1)
        if batch.shape[1] == 1:
          sampled_video['aesthetic'] = torch.concatenate([sampled_video['aesthetic'], sampled_video['aesthetic']], axis=1)
          sampled_video['technical'] = torch.concatenate([sampled_video['technical'], sampled_video['technical']], axis=1)
        return sampled_video, None



    def forward(self, batch, inference=False):
        torch.manual_seed(torch.initial_seed())
        if inference:
            views, _ = spatial_temporal_view_decomposition(batch, self.dopt["sample_types"], self.temporal_samplers)
            for k, v in views.items():
              num_clips = self.dopt["sample_types"][k].get("num_clips", 1)
              views[k] = (
                  ((v.permute(1, 2, 3, 0) - self.mean) / self.std)
                  .permute(3, 0, 1, 2)
                  .reshape(v.shape[0], num_clips, -1, *v.shape[2:])
                  .transpose(0, 1)
                  .to(self.device)
              )
            results = [r.mean().item() for r in self.model(views)]
            return fuse_results(results)
        else:
            views, _ = self.spatial_temporal_view_decomposition_grad(batch)
            for k, v in views.items():
              num_clips = 1
              views[k] = (
                  ((v.permute(1, 2, 3, 0) - self.mean) / self.std)
                  .permute(3, 0, 1, 2)
                  .reshape(v.shape[0], num_clips, -1, *v.shape[2:])
                  .transpose(0, 1)
                  .to(self.device)
              )
            a, b = self.model(views, inference=False)
            loss1 = a.mean()
            loss2 = b.mean()
            loss = (loss1 - 0.1107) / 0.07355 * 0.6104 + (loss2 + 0.08285) / 0.03774 * 0.3896
            loss = 1 / (1 + torch.exp(-loss))
            loss = loss1 + loss2
            return loss
