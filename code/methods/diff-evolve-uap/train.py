#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch.utils.data import DataLoader
import numpy as np

from tqdm import tqdm

from read_dataset import BBDataset
from evaluate import jpeg_generator 

from uap_evaluate import train_main

def score_compressed_images(metric_model, dl_train, noise, sign, metric_range=100, is_fr=False, jpeg_quality=None, device='cpu'):
    losses = 0
    h, w = noise.shape[2:]
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
    with torch.no_grad():
        ds_train = BBDataset(path_gt=path_train, device=device)
        dl_train = DataLoader(ds_train, batch_size=1, shuffle=False)
        sign = -1 if metric_model.lower_better else 1
        t1 = 0.5
        t2 = 0.5
        popsize = 50
        eps = 0.1
        size = (1, 3, 8, 8)
        mut = 0.3
        crossp = 0.5
        max_iter = 200
        pop = torch.rand(popsize, *size).to(device)
        pop_denorm = (2 * pop - 1) * eps
        score = lambda noise: score_compressed_images(metric_model, dl_train, noise, sign, metric_range, is_fr, jpeg_quality, device)
        fitness = torch.asarray([score(ind) for ind in pop_denorm]).float()
        best_idx = torch.argmax(fitness)
        best = (2 * pop[best_idx] - 1) * eps
        for it in tqdm(range(max_iter)):
            for j in range(popsize):
                idxs = [idx for idx in range(popsize) if idx != j]
                try:
                    ft = fitness[idxs]
                    probs = ((ft - ft.min())/(ft.max() - ft.min()) + 1).double().numpy()
                    ids = np.random.choice(idxs, 3, replace = False, p=probs/probs.sum())
                except:
                    ids = np.random.choice(idxs, 3, replace = False)
                a, b, c = pop[ids]
                mut = float(0.3 + 0.7 * torch.rand(1)) if torch.rand(1) < t1 else mut
                mutant = torch.clip(a + mut * (b - c), 0, 1)
                crossp = torch.rand(1) if torch.rand(1) < t2 else crossp
                cross_points = torch.rand(*size) < crossp
                trial = torch.where(cross_points.to(device), mutant, pop[j])
                trial_denorm = (2 * trial - 1) * eps
                f = score(trial_denorm)
                if f < fitness[j]:
                    fitness[j] = f
                    pop[j] = trial
                    if f < fitness[best_idx]:
                        best_idx = j
                        best = trial_denorm
    return best.squeeze().data.cpu().numpy().transpose(1, 2, 0)

if __name__ == "__main__":
    train_main(train)