#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from fgsm_evaluate import test_main

from tqdm import tqdm
import numpy as np

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

from read_dataset import to_numpy, to_torch
import cv2
def compress(img, q):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q]
    np_batch = to_numpy(img)
    if len(np_batch.shape) == 3:
        np_batch = np_batch[np.newaxis]
    jpeg_batch = np.empty(np_batch.shape)
    for i in range(len(np_batch)):
        result, encimg = cv2.imencode('.jpg', np_batch[i] * 255, encode_param)
        jpeg_batch[i] = cv2.imdecode(encimg, 1) / 255
    return torch.nan_to_num(to_torch(jpeg_batch), nan=0)


def score(model, compress_image, ref_image=None, sign=1, q=80):
    if ref_image is not None:
        return sign*model(ref_image, compress(compress_image, q).to(ref_image.device))
    else:
        return sign*model(compress_image)

def attack(compress_image, ref_image=None, model=None, metric_range=100, device='cpu', q=80):
    eps = 0.05
    device = 'cuda'
    model.to(device)
    h, w, c = 256, 256, 3
    n_features = c*h*w
    p_init = 0.05
    n_iters = 10000
    sign = -1 if model.lower_better else 1
    compress_image = compress_image.to(device)
    if ref_image is not None:
        ref_image = ref_image.to(device)

    with torch.no_grad():
        loss = score(model, compress_image, ref_image, sign, q)
        x_best = compress_image.clone()
        for i_iter in tqdm(range(n_iters - 1)):
            x_curr, x_best_curr = compress_image.clone(), x_best.clone()
            deltas = (x_best_curr - x_curr)[..., :h, :w]
            p = p_selection(p_init, i_iter, n_iters)
            s = int(round(np.sqrt(p * n_features / c)))
            s = min(max(s, 1), h-1)
            center_h = np.random.randint(0, h - s)
            center_w = np.random.randint(0, w - s)
            x_curr_window = x_curr[:,:,center_h:center_h+s, center_w:center_w+s]
            x_best_curr_window = x_best_curr[:,:,center_h:center_h+s, center_w:center_w+s]
            while torch.sum(torch.abs(torch.clip(x_curr_window + deltas[:,:,center_h:center_h+s, center_w:center_w+s], 0, 1
                                   ) - x_best_curr_window) < 10**-7) == c*s*s:
                deltas[...,center_h:center_h+s, center_w:center_w+s] = torch.from_numpy(np.random.choice([-eps, eps], size=[c, 1, 1])).to(device)

            deltas = torch.tile(deltas, (1, 1, x_curr.shape[2]//h + 1, x_curr.shape[3]//w + 1))[..., :x_curr.shape[2], :x_curr.shape[3]]
            x_new = torch.clip(x_curr + deltas, 0, 1)

            loss_new = score(model, x_new, ref_image, sign, q)

            if loss < loss_new:
                loss = loss_new
                x_best = x_new

    return compress(x_best, q).cpu().numpy().transpose(0, 2, 3, 1)


if __name__ == "__main__":
    test_main(attack)
