#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from fgsm_evaluate import test_main

from tqdm import tqdm
import numpy as np

import json


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


def logistic_mapping(x, C, S):
    A = 10.0
    B = 0.0
    z = (x - C) / S
    return (A - B) / (1 + torch.exp(-z)) + B


def score(model, compress_image, ref_image=None, q=80, C=1, S=0):
    if ref_image is not None:
        return logistic_mapping(model(compress(compress_image, q).to(ref_image.device), ref_image), C, S)
    else:
        return logistic_mapping(model(compress_image), C, S)


def random_sign(size):
    return torch.sign(-1 + 2 * torch.rand(size=size))


def obj_function(s, mos):
    if mos > 5:
        obj = s
    else:
        obj = -s
    return obj


def attack(
    compress_image,
    ref_image=None,
    model=None,
    metric_range=100,
    device="cpu",
    q=80,
    metric_name=""
):
    eps = 0.05
    n_queries = 10000
    p_init = 0.05

    n_sample = 1
    n_squares = 1

    device = "cuda"
    model.to(device)
    c, h, w = compress_image.shape[1:]
    n_features = c * h * w

    with open("bounds.json") as json_file:
        bounds = json.load(json_file)
        bounds_metric = bounds.get(metric_name, None)
    try:
        C = bounds_metric["high"]
        S = bounds_metric["low"]
    except:
        C = 100
        S = 0


    compress_image = compress_image.to(device)
    if ref_image is not None:
        ref_image = ref_image.to(device)

    init_delta = (eps) * random_sign(size=[compress_image.shape[0], c, 1, w]).to(device)

    compress_image_pert = torch.clamp(compress_image + init_delta, 0, 1)

    fx = score(model, compress_image, ref_image, q, C, S)

    pred_s = score(model, compress_image_pert, ref_image, q, C, S)

    loss_min = obj_function(pred_s, fx)

    x_pert_best = compress_image_pert

    for i_iter in tqdm(range(n_queries - 2)):
        x_pert_curr = x_pert_best.clone()
        for n in range(n_sample):
            deltas_cur = x_pert_curr - compress_image
            p = p_selection(p_init=p_init, it=i_iter, n_iters=n_queries)

            s = int(round(np.sqrt(p * n_features / c)))
            s = min(
                max(s, 1), h - 1, w - 1
            )  # at least c x 1 x 1 window is taken and at most c x h-1 x h-1
            for i in range(n_squares):
                center_h = np.random.randint(0, h - s)
                center_w = np.random.randint(0, w - s)
                x_curr_window = compress_image[
                    0, :, center_h : center_h + s, center_w : center_w + s
                ]
                x_best_curr_window = x_pert_curr[
                    0, :, center_h : center_h + s, center_w : center_w + s
                ]
                # prevent trying out a delta if it doesn't change x_ori (e.g. an overlapping patch)
                while (
                    torch.sum(
                        torch.abs(
                            torch.clamp(
                                x_curr_window
                                + deltas_cur[
                                    0,
                                    :,
                                    center_h : center_h + s,
                                    center_w : center_w + s,
                                ],
                                0,
                                1,
                            )
                            - x_best_curr_window
                        )
                        < 10**-7
                    )
                    == c * s * s
                ):
                    deltas_cur[
                        0, :, center_h : center_h + s, center_w : center_w + s
                    ] = eps * random_sign(size=[c, 1, 1]).to(device)
            x_new = torch.clamp(compress_image + deltas_cur, 0, 1)

            pred_s = score(model, x_new, ref_image, q, C, S)
            loss_candidate = obj_function(pred_s, fx)

            if loss_candidate < loss_min:
                loss_min = loss_candidate.clone()
                x_pert_best = x_new.clone()

    return x_pert_best.cpu().numpy().transpose(0, 2, 3, 1)


if __name__ == "__main__":
    test_main(attack)
