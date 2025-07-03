#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch.utils.data import DataLoader


from tqdm import tqdm

from read_dataset import BBDataset
from evaluate import jpeg_generator
import numpy as np

from uap_evaluate import train_main

def p_selection(p_init, it, n_iters):
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

def score_compressed_images(metric_model, dl_train, noise, sign, metric_range=100, is_fr=False, jpeg_quality=None, device='cpu'):
    losses = 0
    h, w = 256, 256
    for y in dl_train:
        tmp = torch.tile(noise, (1, 1, y.shape[2]//h + 1, y.shape[3]//w + 1))[..., :y.shape[2], :y.shape[3]]
        res = (y + tmp).to(device)
        res.data.clamp_(min=0.,max=1.)
        for orig_image, jpeg_image in jpeg_generator(res, jpeg_quality):
            if is_fr:
                if jpeg_image is None:
                    break
                jpeg_image.data.clamp_(min=0.,max=1.)
                y = y.to(device)
                score = metric_model(y, jpeg_image.to(device))
            else:
                score = metric_model(res)
            if score is None:
                break
            loss = 1 - score.mean() * sign / metric_range
            losses += loss.detach().sum()

    return losses / len(dl_train.dataset)

def train(metric_model, path_train, batch_size=8, is_fr=False, jpeg_quality=None, metric_range=100, device='cpu'):
    eps = 0.1
    h, w, c = 256, 256, 3
    n_features = c*h*w
    p_init = 0.05
    n_iters = 10000
    sign = -1 if metric_model.lower_better else 1
    with torch.no_grad():
        ds_train = BBDataset(path_gt=path_train, device=device)
        dl_train = DataLoader(ds_train, batch_size=1, shuffle=False)
        
        universal_noise = torch.zeros((1, 3, 256, 256)).to(device)
        loss = score_compressed_images(metric_model, dl_train, universal_noise, sign, metric_range, is_fr, jpeg_quality, device)
        
        for i_iter in tqdm(range(n_iters)):
            loss_min_curr = loss.clone()
            deltas_old = universal_noise.clone()
            delta = universal_noise.clone()
        
            p = p_selection(p_init, i_iter, n_iters)
            s = int(round(np.sqrt(p * n_features / c)))
            s = min(max(s, 1), h-1)
            center_h = np.random.randint(0, h - s)
            center_w = np.random.randint(0, w - s)
            while torch.sum(torch.abs(deltas_old[:,:,center_h:center_h+s, center_w:center_w+s] -
                            delta[:,:,center_h:center_h+s, center_w:center_w+s]) < 1e-7) == c*s*s:
                delta[:,:,center_h:center_h+s, center_w:center_w+s] = torch.from_numpy(np.random.choice([-eps, eps], size=[c, 1, 1])).to(delta.device)
            
            loss_new = score_compressed_images(metric_model, dl_train, delta, sign, metric_range, is_fr, jpeg_quality, device)
            if loss > loss_new:
                #print(loss_new)
                loss = loss_new
                universal_noise = delta
    return universal_noise.squeeze().data.cpu().numpy().transpose(1, 2, 0)



if __name__ == "__main__":
    train_main(train)
