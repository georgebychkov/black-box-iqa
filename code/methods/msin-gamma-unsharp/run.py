#!/usr/bin/env python
# -*- coding: utf-8 -*-

from fgsm_evaluate import test_main

import numpy as np
import torchvision.transforms as T
import torch
import random
import albumentations as A
from deap import *
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from torchvision.io import read_image
import numpy as np
import cv2


configs = {'gamma' : {
    'codec': ' -vcodec libx264 -x265-params log-level=error ',
    'mu': 18, 
    'lambda_': 24, 
    'cxpb': 0.5, 
    'mutpb': 0.49, 
    'ngen': 12, 
    'bitrate': 3000, 
    'preset': 'medium', 
    'func': 'gamma', 
    'args_d': {
        'n': 2, 
        'a0_min': 1, 
        'a0_max': 100, 
        'a1_min': 1e-05, 
        'a1_max': 20.0, 
        'type0': 'int', 
        'type1': 'float'
    }
},
'CLAHE' : {
    'codec': ' -vcodec libx264 -x265-params log-level=error ',
    'mu': 18, 
    'lambda_': 24, 
    'cxpb': 0.5, 
    'mutpb': 0.49, 
    'ngen': 12, 
    'bitrate': 3000, 
    'preset': 'medium', 
    'func': 'CLAHE', 
    'args_d': {
        'n': 2, 
        'a0_min': 1, 
        'a0_max': 100, 
        'a1_min': 0.00001, 
        'a1_max': 20.0, 
        'type0': 'int', 
        'type1': 'float'
    }
},
'tonemapDrago' : {
    'mu': 18, 
    'lambda_': 24, 
    'cxpb': 0.5, 
    'mutpb': 0.49, 
    'ngen': 12, 
    'bitrate': 3000, 
    'preset': 'medium', 
    'func': 'tonemapDrago', 
    'args_d': {
            'n': 3,
            'a0_min': 0.0,
            'a0_max': 2.5,
            'a1_min': 0.0,
            'a1_max': 3.0,
            'a2_min': 0.0,
            'a2_max': 1.0,
            'type0': 'float',
            'type1': 'float',
            'type2': 'float'
    }
},


'GammaPlusUnsharp' : {
    'mu': 18, 
    'lambda_': 24, 
    'cxpb': 0.5, 
    'mutpb': 0.49, 
    'ngen': 12, 
    'bitrate': 3000, 
    'preset': 'medium', 
    'func': 'GammaPlusUnsharp', 
    'args_d': {
        'n': 3,
        'a0_min': 30,
        'a0_max': 120,
        'a1_min': 30,
        'a1_max': 120,
        'a2_min': 0.0,
        'a2_max': 2.0,
        'type0': 'int',
        'type1': 'int',
        'type2': 'float'
    }
}

}

config = configs['GammaPlusUnsharp']



from read_dataset import to_numpy, to_torch
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

class calc_met:# 	
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.Results = []	
        
        self.dataset = []
        self.ref_image = []
        self.func = lambda x: x	
        self.sign = -1 if model.lower_better else 1
        self.q = 100
        	
    def get_metrix(self, args):
        frameGT = self.dataset
        ref_image = self.ref_image
        scores = []
        for idx, iarg in enumerate(args):
            try:
                img = self.func(frameGT, *iarg)#GPU_expose_cv2	
            except Exception:
                print("Error in self.func")
                img = 0
                pass
            if ref_image is None:
                scores.append(self.sign*self.model(img))
            else:
                scores.append(self.sign*self.model(ref_image, compress(img, self.q).to(ref_image.device)))
            #print(scores)


        return scores
    
         
def aug4const(img, arg1, sigma, amount):
    img = img.cpu().squeeze().permute((1, 2, 0)).numpy()*255
    arg1 = int(arg1)
    sigma = int(sigma)

        
    light = A.Compose([
        A.RandomGamma(p=1, gamma_limit=(arg1, arg1), always_apply=True),
    ], p=1)
    
    image = light(image = img)['image']
    
    kernel_size = (17, 17)
    
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = cv2.addWeighted(image, (amount + 1.), blurred, - (amount), 0)
    sharpened = np.minimum(sharpened,255)
    sharpened = np.maximum(sharpened,0)
    sharpened = sharpened.round().astype(np.uint8)
    sharpened = sharpened.astype(np.uint8)
    sharpened = torch.from_numpy(sharpened).cuda().permute((2, 0, 1)).unsqueeze(0)/255
    return sharpened

def tonemapDrago(img, arg1, arg2, arg3):
    img = img.cpu().squeeze().permute((1, 2, 0)).numpy()*255

    img = (cv2.createTonemapDrago(arg1, arg2, arg3).process(
        img.astype("float32")/255.)*255).astype("uint8")
    img = torch.from_numpy(img).cuda().permute((2, 0, 1)).unsqueeze(0)/255
    return img

def expose_cv2(frameGT, tilegridsize, cliplimit):
    tilegridsize = int(tilegridsize)
    
    frameGT = frameGT.cpu().squeeze().permute((1, 2, 0)).numpy()*255


    err = np.copy(frameGT)
    ker_cv = cv2.createCLAHE(clipLimit = cliplimit, tileGridSize=( tilegridsize, tilegridsize))
    err[:,:,0] = ker_cv.apply(frameGT[:,:,0].astype("uint8"))
    err[:,:,1] = ker_cv.apply(frameGT[:,:,1].astype("uint8"))
    err[:,:,2] = ker_cv.apply(frameGT[:,:,2].astype("uint8"))


    err = torch.from_numpy(err).cuda().permute((2, 0, 1)).unsqueeze(0)/255
    return err

def Identity(img, arg1, arg2):
    return img

def aug(img, arg1=0, arg2=0):
    img = img.cpu().squeeze().permute((1, 2, 0)).numpy()*255
    light = A.Compose([
        A.RandomGamma(p=1, gamma_limit=(int(arg1), int(arg1)), always_apply=True),
    ], p=1)
    v = light(image=img)
    img = torch.from_numpy(v['image']).cuda().permute((2, 0, 1)).unsqueeze(0)/255
    return img

def init_range(icls):
    global config
    ret_val = []
    for i in range(config['args_d']['n']):
        val1 = float(random.uniform(config['args_d']['a' + str(i) + "_min"] , 
                                    config['args_d']['a' + str(i) + "_max"] )) 
        ret_val.append(val1)
    ind = np.array(ret_val)
    return icls(ind)

def evalOneMax(individual, env):
    individual = np.array(individual)
    res = env.get_metrix(individual)
    return res,

def cxTwoPointCopy(ind1, ind2):
    size = len(ind1)
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else: # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
        = ind2[cxpoint1:cxpoint2].copy(), ind1[cxpoint1:cxpoint2].copy()
    return ind1, ind2

def mut_cutom(individual, indpb = 0.5):
    n_i = random.randint(0, config['args_d']['n'] - 1)
    arg1 = individual[n_i]  + np.random.randn() * (config['args_d']['a' + str(n_i) + "_max"] - config['args_d']['a' + str(n_i) + "_min"]) / 15.
    arg1 = np.clip(arg1, config['args_d']['a' + str(n_i) +  '_min']  , config['args_d']['a' + str(n_i) +  '_max'])
    individual[n_i] = arg1
    return (individual, )

func_map = {
    'GammaPlusUnsharp':aug4const, 
    "CLAHE":expose_cv2, 
    "tonemapDrago":tonemapDrago, 
    "gamma":aug, 
    "I":Identity
    }

def My_eaMuPlusLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen,
                   stats=None, halloffame=None, verbose=__debug__):
    
    statsmy = []
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.evaluate(invalid_ind)
    for ind, fit in zip(invalid_ind, *fitnesses):
        ind.fitness.values = (fit,)
    if halloffame is not None:
        halloffame.update(population)
    for gen in range(1, ngen + 1):
        # Vary the population'
        offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.evaluate(invalid_ind)
        for ind, fit in zip(invalid_ind, *fitnesses):
            ind.fitness.values = (fit,)
        if halloffame is not None:
            halloffame.update(offspring)
        population[:] = toolbox.select(population + offspring, mu)
        log_min = 100
        log_meean = 0
        log_max = 0
        bst_ind = [0, -20]
        for ind in population :
            if ind.fitness.valid:
                ind_value = ind.fitness.values[0]
                if bst_ind[-1] < ind_value:
                    bst_ind = list(ind), ind_value
                log_min = min(log_min, ind_value)
                log_max = max(log_max, ind_value)
                log_meean += ind_value
        log_meean = log_meean / len(population + offspring)
        print(f"GEN: {str(gen)} {log_min} {log_meean} {log_max}")
        
        statsmy.append([population,log_min,log_meean,log_max])
    print(f"BEST IND: {bst_ind}")
    return population, bst_ind, statsmy

    
def attack(compress_image, ref_image=None, model=None, metric_range=100, device='cpu', q=80):
    model = model.to(device)
    compress_image = compress_image.to(device)
    if ref_image is not None:
        ref_image = ref_image.to(device)
    with torch.no_grad():
        env = calc_met(config, model)
        random.seed(a = 431)
        env.func = func_map[config["func"]]
        env.dataset = compress_image
        env.ref_image = ref_image
        env.q = q

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)
        toolbox = base.Toolbox()                 
        toolbox.register("individual", init_range, creator.Individual)
    
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", lambda x: evalOneMax(x, env))
        toolbox.register("mate", cxTwoPointCopy)
        toolbox.register("mutate", mut_cutom, indpb=0.97)
        toolbox.register("select", tools.selBest)
        pop = toolbox.population(n=28)
        hof = tools.HallOfFame(5, similar=np.array_equal)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        pop, logbook, _ = My_eaMuPlusLambda(pop, toolbox ,mu = config['mu'], 
                                                    lambda_ = config['lambda_'], cxpb=config['cxpb'], 
                                                    mutpb=config['mutpb'], ngen=config['ngen'], stats=stats, halloffame=hof)

        adv_image = env.func(compress_image, *logbook[0])
    return compress(adv_image, q)


if __name__ == "__main__":
    test_main(attack)