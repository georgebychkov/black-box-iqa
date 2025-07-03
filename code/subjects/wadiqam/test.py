#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from PIL import Image
from main import RandomCropPatches, NonOverlappingCropPatches, FRnet
import numpy

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
       raise RuntimeException("unsupported resolution")

    device = torch.device("cpu")
    model = FRnet(weighted_average=True).to(device)
    model.load_state_dict(torch.load('checkpoints/WaDIQaM-FR-KADID-10K-EXP1000-5-lr=0.0001-bs=4', map_location='cpu'))
    model.eval()

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

            ref = Image.fromarray(numpy.fromstring(ref, dtype='uint8').reshape((args.height,args.width,bps)), mode="RGB")
            dist = Image.fromarray(numpy.fromstring(dist, dtype='uint8').reshape((args.height,args.width,bps)), mode="RGB")
            data = NonOverlappingCropPatches(dist, ref)
            dist_patches = data[0].unsqueeze(0).to(device)
            ref_patches = data[1].unsqueeze(0).to(device)
            score = model((dist_patches, ref_patches))
            print(score.item())

if __name__ == "__main__":
   main()
