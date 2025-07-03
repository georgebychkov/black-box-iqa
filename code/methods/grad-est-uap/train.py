#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch.utils.data import DataLoader


from tqdm import tqdm

from read_dataset import MyCustomDataset
from evaluate import jpeg_generator
import numpy as np

from uap_evaluate import train_main
from evaluate import compress

def grad_est(metric_model, batch, noise, sigma, N, n, is_fr=False, device='cpu', jpeg_quality=None):
    g = 0
    for i in range(n):
        u = torch.empty((3, N, N)).normal_(mean=0,std=1).to(device)
        img1 = get_adv(batch.to(device), noise + sigma*u)
        img2 = get_adv(batch.to(device), noise - sigma*u)
        if is_fr:
            g += metric_model(batch, compress(img1, jpeg_quality[0], return_torch=True).to(device)).sum()*u
            g -= metric_model(batch, compress(img2, jpeg_quality[0], return_torch=True).to(device)).sum()*u
        else:
            g += metric_model(img1.float()).sum()*u
            g -= metric_model(img2.float()).sum()*u
    return 1/(2*n*sigma*len(batch))*g
    
def score_images(metric_model, noise, ds_train):
    dl_train = DataLoader(ds_train, batch_size=3, shuffle=False)
    loss = []
    ds_iter = iter(dl_train)
    for _ in range(len(dl_train)):
        y = next(ds_iter)
        loss.append(metric_model(get_adv(y, noise)).cpu().numpy().ravel())
    return np.mean(np.hstack(loss))
    

def get_adv(im, pert):
    return torch.clamp(im + torch.tile(pert, (1, 1, im.shape[2] // pert.shape[2]+ 1, im.shape[3] // pert.shape[3]+ 1))[
        ...,:im.shape[2], :im.shape[3]], 0, 1)

def train(metric_model, path_train, batch_size=8, is_fr=False, jpeg_quality=None, metric_range=100, device='cpu'):
    eps = 0.1
    sigma = 0.001
    N = 32
    n = 20
    eta = 0.01
    max_iters = 250
    batch = 5
    sign = -1 if metric_model.lower_better else 1
    with torch.no_grad():
        ds_train = MyCustomDataset(path_gt=path_train, device=device)
        dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
        universal_noise = torch.zeros((1, 3, N, N)).to(device)
        for i_iter in tqdm(range(max_iters)):
            ds_iter = iter(dl_train)
            for _ in range(len(dl_train)):
                y = next(ds_iter)
                g = grad_est(metric_model, y, universal_noise, sigma, N, n, is_fr, device, jpeg_quality)
                g = g/torch.linalg.norm(g)
                universal_noise += eta * g * sign
                universal_noise = torch.clip(universal_noise, -eps, eps)

        
    return universal_noise.squeeze().data.cpu().numpy().transpose(1, 2, 0)



if __name__ == "__main__":
    train_main(train)