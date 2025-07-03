#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm

from read_dataset import BBDataset
from evaluate import jpeg_generator
import numpy as np
import json

from uap_evaluate import train_main

def score_compressed_images(metric_model, dl_train, noise, metric_range=100, is_fr=False, jpeg_quality=None, device='cpu', C=1, S=0):
    losses = 0
    h, w = 256, 256
    for y in dl_train:
        for orig_image, jpeg_image in jpeg_generator(y, jpeg_quality):
            if is_fr:
                if jpeg_image is None:
                    break
                tmp = torch.tile(noise, (1, 1, y.shape[2]//h + 1, y.shape[3]//w + 1))[..., :y.shape[2], :y.shape[3]]
                res = (jpeg_image.to(device) + tmp).type(torch.FloatTensor).to(device)
                res.data.clamp_(min=0.,max=1.)
                y = y.to(device)
                score = metric_model(y, res)
            else:
                tmp = torch.tile(noise, (1, 1, y.shape[2]//h + 1, y.shape[3]//w + 1))[..., :y.shape[2], :y.shape[3]]
                
                res = (y + tmp).to(device)
                res.data.clamp_(min=0.,max=1.)
                score = metric_model(res)

            if score is None:
                break
            loss = logistic_mapping(score, C, S)
            
            losses += loss.detach().sum()

    return losses / len(dl_train.dataset)    

def p_selection(p_init, it, n_iters):
    """Piece-wise constant schedule for p (the fraction of pixels changed on every iteration)."""
    it = int(it / n_iters * 10000)

    if 10 < it <= 50:
        p = p_init / 2
    elif 50 < it <= 200:
        p = p_init / 4
    elif 200 < it <= 500:
        p = p_init / 8
    elif 500 < it <= 1000:
        p = p_init / 16
    elif 1000 < it <= 2000:
        p = p_init / 32
    elif 2000 < it <= 4000:
        p = p_init / 64
    elif 4000 < it <= 6000:
        p = p_init / 128
    elif 6000 < it <= 8000:
        p = p_init / 256
    elif 8000 < it <= 10000:
        p = p_init / 512
    else:
        p = p_init

    return p

def logistic_mapping(x, C, S):
    A = 10.0
    B = 0.0
    z = (x - C) / S
    return (A - B) / (1 + torch.exp(-z)) + B


def random_sign(size):
    return torch.sign(-1 + 2 * torch.rand(size=size))


def obj_function(s, mos):
    if mos > 5:
        obj = s
    else:
        obj = -s
    return obj


def train(metric_model, path_train, batch_size=8, is_fr=False, jpeg_quality=None, metric_range=100, device='cpu',  metric_name=""):
    
    eps = 0.1
    n_queries = 10000
    p_init = 0.05

    n_sample = 1
    n_squares = 1

    device = "cuda"
    metric_model.to(device)
    h, w, c = 256, 256, 3
    n_features = c * h * w

    with open("bounds.json") as json_file:
        bounds = json.load(json_file)
        bounds_metric = bounds.get(metric_name, None)
    try:
        C = bounds_metric["high"]
        S = bounds_metric["low"]
        if S == 0:
            S = 1e-10
    except:
        C = 100
        S = 1

    ds_train = BBDataset(path_gt=path_train, device=device)
    dl_train = DataLoader(ds_train, batch_size=1, shuffle=False)
        
    universal_noise = torch.zeros((1, 3, 256, 256)).to(device)
    #loss = score_compressed_images(metric_model, dl_train, universal_noise, metric_range, is_fr, jpeg_quality, device, C, S)

    init_delta = (eps) * random_sign(size=[1, c, 1, w]).to(device)
    
    init_delta = universal_noise + init_delta

    fx = score_compressed_images(metric_model, dl_train, universal_noise, metric_range, is_fr, jpeg_quality, device, C, S)

    pred_s = score_compressed_images(metric_model, dl_train, init_delta, metric_range, is_fr, jpeg_quality, device, C, S)

    loss_min = obj_function(pred_s, fx)

    universal_noise = init_delta.clone()

    for i_iter in tqdm(range(n_queries - 2)):
        for n in range(n_sample):
            deltas_cur = universal_noise.clone()
            delta = universal_noise.clone()
            p = p_selection(p_init=p_init, it=i_iter, n_iters=n_queries)

            s = int(round(np.sqrt(p * n_features / c)))
            s = min(
                max(s, 1), h - 1, w - 1
            )  # at least c x 1 x 1 window is taken and at most c x h-1 x h-1
            for i in range(n_squares):
                center_h = np.random.randint(0, h - s)
                center_w = np.random.randint(0, w - s)
                # prevent trying out a delta if it doesn't change x_ori (e.g. an overlapping patch)
                while (
                    torch.sum(
                        torch.abs(
                            deltas_cur[
                                    0,
                                    :,
                                    center_h : center_h + s,
                                    center_w : center_w + s,
                                ]
                            - delta[
                                    0,
                                    :,
                                    center_h : center_h + s,
                                    center_w : center_w + s,
                                ]
                        )
                        < 10**-7
                    )
                    == c * s * s
                ):
                    deltas_cur[
                        0, :, center_h : center_h + s, center_w : center_w + s
                    ] = eps * random_sign(size=[1, c, 1, 1]).to(device)

            pred_s = score_compressed_images(metric_model, dl_train, deltas_cur, metric_range, is_fr, jpeg_quality, device, C, S)
            loss_candidate = obj_function(pred_s, fx)
      
            if loss_candidate < loss_min:
                print(loss_candidate)
                loss_min = loss_candidate.clone()
                universal_noise = deltas_cur.clone()

    return universal_noise.squeeze().data.cpu().numpy().transpose(1, 2, 0)
      


if __name__ == "__main__":
    with torch.no_grad():
        train_main(train)
