#!/usr/bin/env python
# -*- coding: utf-8 -*-

from fgsm_evaluate import test_main

import math
import numpy as np
import torch

import time
import math

import numpy as np
from copy import deepcopy

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

def get_score(model, x, y, q):
    sign = -1 if model.lower_better else 1
    if y is None:
        return sign*model(x)
    return sign*model(y, compress(x, q).to(y.device))

class RSAttack():
    """
    Sparse-RS attacks

    :param predict:           forward pass function
    :param norm:              type of the attack
    :param n_restarts:        number of random restarts
    :param n_queries:         max number of queries (each restart)
    :param eps:               bound on the sparsity of perturbations
    :param seed:              random seed for the starting point
    :param alpha_init:        parameter to control alphai
    :param loss:              loss function optimized ('margin', 'ce' supported)
    :param resc_schedule      adapt schedule of alphai to n_queries
    :param device             specify device to use
    :param log_path           path to save logfile.txt
    :param constant_schedule  use constant alphai
    :param targeted           perform targeted attacks
    :param init_patches       initialization for patches
    :param resample_loc       period in queries of resampling images and
                              locations for universal attacks
    :param data_loader        loader to get new images for resampling
    :param update_loc_period  period in queries of updates of the location
                              for image-specific patches
    """
    
    def __init__(
            self,
            model,
            norm='L0',
            n_queries=5000,
            eps=150,
            p_init=.8,
            n_restarts=1,
            seed=0,
            resc_schedule=True,
            constant_schedule=False,
            init_patches='random_squares',
            q=80):
        """
        Sparse-RS implementation in PyTorch
        """
        
        self.model = model
        self.norm = norm
        self.n_queries = n_queries
        self.eps = eps
        self.p_init = p_init
        self.n_restarts = n_restarts
        self.seed = seed
        self.rescale_schedule = resc_schedule
        self.constant_schedule = constant_schedule
        self.init_patches = init_patches
        self.resample_loc = n_queries // 10
        self.update_loc_period = 4
        self.device = 'cuda'
        self.q = q

    def init_hyperparam(self, x):
        assert self.norm in ['L0', 'patches', 'frames',
            'patches_universal', 'frames_universal']

        if self.device is None:
            self.device = x.device
        if self.seed is None:
            self.seed = time.time()
        if 'universal' not in self.norm:
            self.orig_dim = list(x.shape[1:])
            self.ndims = len(self.orig_dim)


    def check_shape(self, x):
        return x if len(x.shape) == (self.ndims + 1) else x.unsqueeze(0)

    def random_choice(self, shape):
        t = 2 * torch.rand(shape).to(self.device) - 1
        return torch.sign(t)

    def random_int(self, low=0, high=1, shape=[1]):
        t = low + (high - low) * torch.rand(shape).to(self.device)
        return t.long()

    def normalize(self, x):
        if self.norm == 'Linf':
            t = x.abs().view(x.shape[0], -1).max(1)[0]
            return x / (t.view(-1, *([1] * self.ndims)) + 1e-12)

        elif self.norm == 'L2':
            t = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()
            return x / (t.view(-1, *([1] * self.ndims)) + 1e-12)

    def lp_norm(self, x):
        if self.norm == 'L2':
            t = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()
            return t.view(-1, *([1] * self.ndims))

    def p_selection(self, it):
        """ schedule to decrease the parameter p """

        if self.rescale_schedule:
            it = int(it / self.n_queries * 10000)

        if 'patches' in self.norm:
            if 10 < it <= 50:
                p = self.p_init / 2
            elif 50 < it <= 200:
                p = self.p_init / 4
            elif 200 < it <= 500:
                p = self.p_init / 8
            elif 500 < it <= 1000:
                p = self.p_init / 16
            elif 1000 < it <= 2000:
                p = self.p_init / 32
            elif 2000 < it <= 4000:
                p = self.p_init / 64
            elif 4000 < it <= 6000:
                p = self.p_init / 128
            elif 6000 < it <= 8000:
                p = self.p_init / 256
            elif 8000 < it:
                p = self.p_init / 512
            else:
                p = self.p_init

        elif 'frames' in self.norm:
            if not 'universal' in self.norm :
                tot_qr = 10000 if self.rescale_schedule else self.n_queries
                p = max((float(tot_qr - it) / tot_qr  - .5) * self.p_init * self.eps ** 2, 0.)
                return 3. * math.ceil(p)
        
            else:
                assert self.rescale_schedule
                its = [200, 600, 1200, 1800, 2500, 10000, 100000]
                resc_factors = [1., .8, .6, .4, .2, .1, 0.]
                c = 0
                while it >= its[c]:
                    c += 1
                return resc_factors[c] * self.p_init
        
        elif 'L0' in self.norm:
            if 0 < it <= 50:
                p = self.p_init / 2
            elif 50 < it <= 200:
                p = self.p_init / 4
            elif 200 < it <= 500:
                p = self.p_init / 5
            elif 500 < it <= 1000:
                p = self.p_init / 6
            elif 1000 < it <= 2000:
                p = self.p_init / 8
            elif 2000 < it <= 4000:
                p = self.p_init / 10
            elif 4000 < it <= 6000:
                p = self.p_init / 12
            elif 6000 < it <= 8000:
                p = self.p_init / 15
            elif 8000 < it:
                p = self.p_init / 20
            else:
                p = self.p_init
        
            if self.constant_schedule:
                p = self.p_init / 2
        
        return p

    def sh_selection(self, it):
        """ schedule to decrease the parameter p """

        t = max((float(self.n_queries - it) / self.n_queries - .0) ** 1., 0) * .75

        return t
    
    def get_init_patch(self, c, s, n_iter=1000):
        if self.init_patches == 'stripes':
            patch_univ = torch.zeros([1, c, s, s]).to(self.device) + self.random_choice(
                [1, c, 1, s]).clamp(0., 1.)
        elif self.init_patches == 'uniform':
            patch_univ = torch.zeros([1, s, s, c]).to(self.device) + self.random_choice(
                [1, c, 1, 1]).clamp(0., 1.)
        elif self.init_patches == 'random':
            patch_univ = self.random_choice([1, c, s, s]).clamp(0., 1.)
        elif self.init_patches == 'random_squares':
            patch_univ = torch.zeros([1, c, s, s]).to(self.device)
            for _ in range(n_iter):
                size_init = torch.randint(low=1, high=math.ceil(s ** .5), size=[1]).item()
                loc_init = torch.randint(s - size_init + 1, size=[2])
                patch_univ[0, :, loc_init[0]:loc_init[0] + size_init, loc_init[1]:loc_init[1] + size_init] = 0.
                patch_univ[0, :, loc_init[0]:loc_init[0] + size_init, loc_init[1]:loc_init[1] + size_init
                    ] += self.random_choice([c, 1, 1]).clamp(0., 1.)
        elif self.init_patches == 'sh':
            patch_univ = torch.ones([1, c, s, s]).to(self.device)
        
        return patch_univ.clamp(0., 1.)
    
    def attack_single_run(self, x, y):
        with torch.no_grad():
            c, h, w = x.shape[1:]
            
            if self.norm == 'L0':
                eps = self.eps
                x_best = x.clone()
                n_pixels = h * w
                b_all, be_all = torch.zeros([x.shape[0], eps]).long(), torch.zeros([x.shape[0], n_pixels - eps]).long()
                for img in range(x.shape[0]):
                    ind_all = torch.randperm(n_pixels)
                    ind_p = ind_all[:eps]
                    ind_np = ind_all[eps:]
                    x_best[img, :, ind_p // w, ind_p % w] = self.random_choice([c, eps]).clamp(0., 1.)
                    b_all[img] = ind_p.clone()
                    be_all[img] = ind_np.clone()
                    
                loss_min = get_score(self.model, x_best, y, self.q)
                #print(loss_min)
                n_queries = torch.ones(x.shape[0]).to(self.device)
                
                for it in range(1, self.n_queries):
                    # check points still to fool
                    x_curr = self.check_shape(x)
                    x_best_curr = self.check_shape(x_best)
                    loss_min_curr = loss_min.clone()
                    b_curr, be_curr = b_all.clone(), be_all.clone()
                    
                    # build new candidate
                    x_new = x_best_curr.clone()
                    eps_it = max(int(self.p_selection(it) * eps), 1)
                    ind_p = torch.randperm(eps)[:eps_it]
                    ind_np = torch.randperm(n_pixels - eps)[:eps_it]
                    
                    for img in range(x_new.shape[0]):
                        p_set = b_curr[img, ind_p]
                        np_set = be_curr[img, ind_np]
                        x_new[img, :, p_set // w, p_set % w] = x_curr[img, :, p_set // w, p_set % w].clone()
                        if eps_it > 1:
                            x_new[img, :, np_set // w, np_set % w] = self.random_choice([c, eps_it]).clamp(0., 1.)
                        else:
                            # if update is 1x1 make sure the sampled color is different from the current one
                            old_clr = x_new[img, :, np_set // w, np_set % w].clone()
                            assert old_clr.shape == (c, 1), print(old_clr)
                            new_clr = old_clr.clone()
                            while (new_clr == old_clr).all().item():
                                new_clr = self.random_choice([c, 1]).clone().clamp(0., 1.)
                            x_new[img, :, np_set // w, np_set % w] = new_clr.clone()
                        
                    # compute loss of the new candidates
                    loss = get_score(self.model, x_new, y, self.q)
                    n_queries += 1
                    
                    # update best solution
                    if loss > loss_min_curr:
                        loss_min = loss
                        nimpr = 1
                    else:
                        nimpr = 0
                    if nimpr > 0.:
                        #print(loss)
                        x_best = x_new.clone()
                        t = b_curr.clone()
                        te = be_curr.clone()
                        t[:, ind_p] = be_curr[:,ind_np] + 0
                        te[:, ind_np] = b_curr[:,ind_p] + 0
                        
                        b_all = t.clone()
                        be_all = te.clone()
        
        return n_queries, x_best

    def perturb(self, x, y):
        """
        :param x:           clean images
        :param y:           untargeted attack -> clean labels,
                            if None we use the predicted labels
                            targeted attack -> target labels, if None random classes,
                            different from the predicted ones, are sampled
        """

        self.init_hyperparam(x)
        adv = x.clone()
        qr = torch.zeros([len(x)]).to(self.device)

        torch.random.manual_seed(self.seed)
        torch.cuda.random.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        for counter in range(self.n_restarts):
            x_to_fool = x.clone()

            qr_curr, adv_curr = self.attack_single_run(x_to_fool, y)
            
            adv = adv_curr.clone()
            qr = qr_curr.clone()


        return qr, adv

    
def attack(compress_image, ref_image=None, model=None, metric_range=100, device='cpu', q=80):
    max_queries = 10000
    eps = 169
    device='cuda'
    with torch.no_grad():
      model = model.to(device)
      compress_image = compress_image.to(device)
      if ref_image is not None:
          ref_image = ref_image.to(device)
      attack = RSAttack(model, eps=eps, norm='L0', n_queries=max_queries, q=q)

      res = attack.perturb(compress_image, ref_image)
    return compress(res[1], q)

if __name__ == "__main__":
    test_main(attack)