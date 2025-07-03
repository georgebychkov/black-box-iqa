#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
import torch
from torchvision import transforms
import numpy as np
import pyiqa

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist_path", type=str)
    parser.add_argument("--ref_path", type=str)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    args = parser.parse_args()
    bps = 3
    if args.width * args.height <= 0:
       raise RuntimeError("unsupported resolution")

    model = pyiqa.create_metric(
        'ahiq',
        as_loss=True,
        pretrained_model_path='AHIQ_vit_p8_epoch33-da3ea303.pth',
        checkpoint_resnet='resnet50_a1_0-14fe96d1.pth',
        checkpoint_vit='B_8-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz',
        device='cuda:0'
    )
    #model.training = True
    
    transform = transforms.ToTensor()

    
    print("value")
    with open(args.ref_path, 'rb') as ref_rgb24, open(args.dist_path, 'rb') as dist_rgb24, torch.no_grad():
        while True:
            ref = ref_rgb24.read(args.width * args.height * bps)
            dist = dist_rgb24.read(args.width * args.height * bps)
            
            if len(ref) == 0 and len(dist) == 0:
                break
            if len(ref) != args.width * args.height * bps:
                raise RuntimeError("unexpected end of stream ref_path")
            if len(dist) != args.width * args.height * bps:
                raise RuntimeError("unexpected end of stream dist_path")
                
            ref = np.frombuffer(ref, dtype='uint8').reshape((args.height,args.width,bps))
            dist = np.frombuffer(dist, dtype='uint8').reshape((args.height,args.width,bps))
            
            score = -torch.mean(model(torch.unsqueeze(transform(ref), 0), torch.unsqueeze(transform(dist), 0))).item()
            print(score)

if __name__ == "__main__":
   main()
