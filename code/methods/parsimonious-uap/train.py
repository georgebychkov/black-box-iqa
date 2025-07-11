#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch.utils.data import DataLoader

import itertools
import math
import numpy as np
import heapq
import torchvision.transforms as T
from tqdm import tqdm

from read_dataset import BBDataset
from evaluate import jpeg_generator
import numpy as np

from uap_evaluate import train_main

def score_compressed_images(metric_model, dl_train, noise, metric_range=100, is_fr=False, jpeg_quality=None, device='cpu'):
    losses = 0
    h, w = 256, 256
    sign = -1 if metric_model.lower_better else 1
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

    return -losses / len(dl_train.dataset)


class LocalSearchHelper(object):
  """A helper for local search algorithm.
  Note that since heapq library only supports min heap, we flip the sign of loss function.
  """

  def __init__(self, model, epsilon, max_iters):
    """Initalize local search helper.
    
    Args:
      model: model
      loss_func: str, the type of loss function
      epsilon: float, the maximum perturbation of pixel value
    """
    # Hyperparameter setting 
    self.epsilon = epsilon
    self.max_iters = max_iters
    self.model = model

  def _flip_noise(self, noise, block):
    """Flip the sign of perturbation on a block.
    Args:
      noise: numpy array of size [3, 256, 256, 3], a noise
      block: [upper_left, lower_right, channel], a block
    
    Returns:
      noise: numpy array of size [3, 256, 256], an updated noise 
    """
    noise_new = noise.clone()
    upper_left, lower_right, channel = block 
    noise_new[:, channel, upper_left[0]:lower_right[0], upper_left[1]:lower_right[1]] *= -1
    return noise_new

  def perturb(self, dl_train, noise, blocks, is_fr=False, jpeg_quality=None, metric_range=100,
               device='cpu'):
    """Update a noise with local search algorithm.
    
    Args:
      image: numpy array of size [3, 299, 299], an original image
      noise: numpy array of size [3, 256, 256], a noise
      blocks: list, a set of blocks

    Returns: 
      noise: numpy array of size [3, 256, 256], an updated noise
      num_queries: int, the number of queries
      curr_loss: float, the value of loss function
    """
    device = noise.device
    # Class variables
    # Local variables
    priority_queue = []
    num_queries = 0
    
    # Check if a block is in the working set or not
    A = np.zeros([len(blocks)], np.int32)
    for i, block in enumerate(blocks):
      upper_left, _, channel = block
      x = upper_left[0]
      y = upper_left[1]
      # If the sign of perturbation on the block is positive,
      # which means the block is in the working set, then set A to 1
      if noise[0, channel, x, y] > 0:
        A[i] = 1

    # Calculate the current loss  
    loss = score_compressed_images(self.model, dl_train, noise[0], metric_range, is_fr,
                                    jpeg_quality, device)
    #print(loss)
    num_queries += 1
    curr_loss = loss
  
    # Main loop
    for _ in range(self.max_iters):
      # Lazy greedy insert
      indices,  = np.where(A==0)
      
      batch_size = 100
      num_batches = int(math.ceil(len(indices)/batch_size))
      
      for ibatch in range(num_batches):
        bstart = ibatch * batch_size
        bend = min(bstart + batch_size, len(indices))
        
        noise_batch = torch.zeros([bend-bstart, 3, 256, 256]).to(noise.device)
         
        for i, idx in enumerate(indices[bstart:bend]):
          noise_batch[i, ...] = self._flip_noise(noise, blocks[idx])
        
        # Early stopping 
        num_queries += bend-bstart

        # Push into the priority queue
        for i in range(bend-bstart):
          losses = score_compressed_images(self.model, dl_train, noise_batch[
             i, ...], metric_range, is_fr, jpeg_quality, device)
          #print(losses)
          idx = indices[bstart+i]
          margin = -(losses-curr_loss)
          heapq.heappush(priority_queue, (margin, idx))
      
      # Pick the best element and insert it into the working set   
      if len(priority_queue) > 0:
        best_margin, best_idx = heapq.heappop(priority_queue)
        curr_loss += best_margin
        noise = self._flip_noise(noise, blocks[best_idx])
        A[best_idx] = 1
      
      # Add elements into the working set
      while len(priority_queue) > 0:
        # Pick the best element
        cand_margin, cand_idx = heapq.heappop(priority_queue)
        
        # Re-evalulate the element
        losses = score_compressed_images(self.model, dl_train, self._flip_noise(
           noise, blocks[cand_idx]), metric_range, is_fr, jpeg_quality, device)
        #print(losses)
        num_queries += 1
        margin = -(losses-curr_loss)
        
        # If the cardinality has not changed, add the element
        if len(priority_queue) == 0 or margin <= priority_queue[0][0]:
          # If there is no element that has negative margin, then break
          if margin > 0:
            break
          # Update the noise
          curr_loss = losses
          noise = self._flip_noise(noise, blocks[cand_idx])
          A[cand_idx] = 1
          # Early stopping
        # If the cardinality has changed, push the element into the priority queue
        else:
          heapq.heappush(priority_queue, (margin, cand_idx))

      priority_queue = []

      # Lazy greedy delete
      indices,  = np.where(A==1)
       
      batch_size = 100
      num_batches = int(math.ceil(len(indices)/batch_size))   
      
      for ibatch in range(num_batches):
        bstart = ibatch * batch_size
        bend = min(bstart + batch_size, len(indices))
        
        noise_batch = torch.zeros([bend-bstart, 3, 256, 256]).to(noise.device)
        
        for i, idx in enumerate(indices[bstart:bend]):
          noise_batch[i, ...] = self._flip_noise(noise, blocks[idx])
        
        # Early stopping
        num_queries += bend-bstart

        # Push into the priority queue
        for i in range(bend-bstart):
          losses = score_compressed_images(self.model, dl_train, noise_batch[
             i, ...], metric_range, is_fr, jpeg_quality, device)
          #print(losses)
          idx = indices[bstart+i]
          margin = -(losses-curr_loss)
          heapq.heappush(priority_queue, (margin, idx))

      # Pick the best element and remove it from the working set   
      if len(priority_queue) > 0:
        best_margin, best_idx = heapq.heappop(priority_queue)
        curr_loss += best_margin
        noise = self._flip_noise(noise, blocks[best_idx])
        A[best_idx] = 0
      
      # Delete elements into the working set
      while len(priority_queue) > 0:
        # pick the best element
        cand_margin, cand_idx = heapq.heappop(priority_queue)
        
        # Re-evalulate the element        
        losses = score_compressed_images(self.model, dl_train, self._flip_noise(
           noise, blocks[cand_idx]), metric_range, is_fr, jpeg_quality, device)
        #print(losses)
        num_queries += 1 
        margin = -(losses-curr_loss)
      
        # If the cardinality has not changed, remove the element
        if len(priority_queue) == 0 or margin <= priority_queue[0][0]:
          # If there is no element that has negative margin, then break
          if margin >= 0:
            break
          # Update the noise
          curr_loss = losses
          noise = self._flip_noise(noise, blocks[cand_idx])
          A[cand_idx] = 0
          # Early stopping
        # If the cardinality has changed, push the element into the priority queue
        else:
          heapq.heappush(priority_queue, (margin, cand_idx))
      
      priority_queue = []
    
    return noise, num_queries, curr_loss


def split_block(upper_left, lower_right, block_size):
    """Split an image into a set of blocks. 
    Note that a block consists of [upper_left, lower_right, channel]
    
    Args:
      upper_left: [x, y], the coordinate of the upper left of an image
      lower_right: [x, y], the coordinate of the lower right of an image
      block_size: int, the size of a block

    Returns:
      blocks: list, the set of blocks
    """
    blocks = []
    xs = np.arange(upper_left[0], lower_right[0], block_size)
    ys = np.arange(upper_left[1], lower_right[1], block_size)
    for x, y in itertools.product(xs, ys):
        for c in range(3):
            blocks.append([[x, y], [x+block_size, y+block_size], c])
    return blocks

def perturb_image(image, noise):
    """Given an image and a noise, generate a perturbed image. 
    First, resize the noise with the size of the image. 
    Then, add the resized noise to the image.

    Args:
      image: torch tensor of size [3, 299, 299], an original image
      noise: torch tensor of size [3, 256, 256], a noise
      
    Returns:
      adv_iamge: torch tensor of size [3, 299, 299], an perturbed image   
    """
    adv_image = image + T.Resize(size = (image.shape[2], image.shape[3]), interpolation=T.InterpolationMode.NEAREST)(noise)
    adv_image = torch.clip(adv_image, 0., 1.)
    return adv_image
    

def train(metric_model, path_train, batch_size=8, is_fr=False, jpeg_quality=None, metric_range=100, device='cpu'):
    max_queries = 10000
    epsilon = 0.1
    b_s = 64
    block_size = 32
    no_hier = False
    max_iters = 1
    # Local variables
    num_queries = 0
    upper_left = [0, 0]
    lower_right = [256, 256]
    blocks = split_block(upper_left, lower_right, block_size)
    
    # Initialize a noise to -epsilon
    noise = -epsilon*torch.ones((1, 3, 256, 256)).to(device)

    # Construct a batch
    num_blocks = len(blocks)
    batch_size = batch_size if batch_size > 0 else num_blocks
    curr_order = np.random.permutation(num_blocks)

    
    local_search = LocalSearchHelper(metric_model, epsilon, max_iters)
    with torch.no_grad():
      ds_train = BBDataset(path_gt=path_train, device=device)
      dl_train = DataLoader(ds_train, batch_size=1, shuffle=False)
      # Main loop
      while True:
        # Run batch
        num_batches = int(math.ceil(num_blocks/b_s))
        for i in range(num_batches):
          # Pick a mini-batch
          bstart = i*b_s
          bend = min(bstart + b_s, num_blocks)
          blocks_batch = [blocks[curr_order[idx]] for idx in range(bstart, bend)]
          # Run local search algorithm on the mini-batch
          noise, queries, loss = local_search.perturb(
            dl_train, noise, blocks_batch, is_fr, jpeg_quality, metric_range, device)
          
          num_queries += queries
          # If query count exceeds the maximum queries, then return False
          if num_queries > max_queries:
            return noise.squeeze().data.cpu().numpy().transpose(1, 2, 0)
          # Generate an adversarial image
      
        # If block size >= 2, then split the iamge into smaller blocks and reconstruct a batch
        if not no_hier and block_size >= 2:
          block_size //= 2
          blocks = split_block(upper_left, lower_right, block_size)
          num_blocks = len(blocks)
          batch_size = batch_size if batch_size > 0 else num_blocks
          curr_order = np.random.permutation(num_blocks)
        # Otherwise, shuffle the order of the batch
        else:
          curr_order = np.random.permutation(num_blocks)


if __name__ == "__main__":
    train_main(train)
