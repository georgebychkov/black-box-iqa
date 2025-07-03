#!/usr/bin/env python
# -*- coding: utf-8 -*-


import torch
import numpy as np
from torchvision import transforms
from run_main import get_model
from data.sampling.patch_sampling import get_iqa_patches, PatchSampler

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
    
    model = get_model('cuda:0', True, 'latest.pth')

    transform = transforms.ToTensor()
    sampler = PatchSampler()
    tensr2PIL = transforms.ToPILImage()
    
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
            
            
            ref_tensor = transform(ref)
            dist_tensor = transform(dist)
            # Last parameter changed to 1! It was 16 but was throwing an error
            patches = get_iqa_patches([tensr2PIL(ref_tensor), tensr2PIL(dist_tensor)], [ref_tensor,dist_tensor],16,(16,16), sampler, 1)  
            score = -torch.mean(model([patches[0].unsqueeze(0).to('cuda:0'), patches[1].unsqueeze(0).to('cuda:0')], patches_scales=patches[3].to('cuda:0'), patches_pos=patches[2].to('cuda:0'))).item()
            print(score)

if __name__ == "__main__":
   main()