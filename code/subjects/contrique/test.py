#!/usr/bin/env python
# -*- coding: utf-8 -*-


import torch
import numpy as np
from torchvision.transforms import Resize, ToTensor
import torchvision
import pickle
from src.modules.CONTRIQUE_model import CONTRIQUE_model


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
    
    encoder = torchvision.models.resnet50()
    encoder.load_state_dict(torch.load('resnet50-0676ba61.pth'))

    model = CONTRIQUE_model(None, encoder, 2048)
    model.load_state_dict(torch.load('CONTRIQUE_checkpoint25.tar', map_location='cuda'))
    model.to('cuda')
    model.eval()
    
    regressor = pickle.load(open('src/models/CSIQ_FR.save', 'rb'))

    
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
            
            ref = ToTensor()(ref).unsqueeze(0).to('cuda')
            dist = ToTensor()(dist).unsqueeze(0).to('cuda')
            
            ref_res = Resize((args.height // 2, args.width // 2))(ref)
            dist_res = Resize((args.height // 2, args.width // 2))(dist)

            _,_, _, _, ref_feat, ref_feat_2, _, _ = model(ref, ref_res)
            _,_, _, _, dist_feat, dist_feat_2, _, _ = model(dist, dist_res)
            
            ref = torch.hstack((ref_feat, ref_feat_2))
            dist = torch.hstack((dist_feat ,dist_feat_2))
            feat = torch.abs(ref - dist)

            score = regressor.predict(feat.cpu())[0]
            print(score)

if __name__ == "__main__":
   main()