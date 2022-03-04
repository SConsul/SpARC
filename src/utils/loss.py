import torch
import torch.nn.functional as F
import numpy as np


def binary_sim_loss(batch):
    batch = F.normalize(batch,dim=1) #shape (B,L)
    dot_prods = batch@batch.T #shape (B,B), stores all the dot products of every combination
    loss = (torch.sum(dot_prods))/2.0
    for i in range(0,len(batch),2):
        loss -= 2*dot_prods[i][i+1]

    return loss