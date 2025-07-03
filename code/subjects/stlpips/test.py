#!/usr/bin/env python
# -*- coding: utf-8 -*-


import torch
import numpy as np
from torchvision import transforms
from stlpips_pytorch import  stlpips

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
    
    model = stlpips.LPIPS(net='alex', variant="shift_tolerant", pretrained=True,  model_path='alex_shift_tolerant.pth').to('cuda:0')

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
            
            score = -torch.mean(model(torch.unsqueeze(transform(ref).to('cuda:0'), 0), torch.unsqueeze(transform(dist).to('cuda:0'), 0))).item()
            print(score)

if __name__ == "__main__":
   main()