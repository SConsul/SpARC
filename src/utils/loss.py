import torch
import torch.nn.functional as F
import numpy as np


def binary_sim_loss(batch,idx):
    """
    Computes the similarity/difference loss:
    ∑[i = 0, 2, ..., 2B] (1 - a_i.a_i) + ∑[i=0,1,...]∑[j!=i,j!=i+1] a_i . a_j
    First sum is -ve cosine similarity of similar questions,
    second sum is cosine similarity of each question with every other dissimilar question.
    :param batch: shape (2B, L, C) of (a_0, a_0', a_1, a_1', ..., a_B, a_B')
        where consecutive questions are similar
    :return: Loss value
    """
    if idx is not None:
        idx = idx.view(-1) #shape 2B,
        r = torch.LongTensor(range(idx.shape[0]))
        batch = batch[r,idx[r],:] #shape = (2B, C)
    else:
        b, L, c = batch.shape
        batch = batch.view(b, -1) #shape = (2B, C)

    # Unit norm vectors
    batch = F.normalize(batch, dim=1)  # shape (B,L*C)

    # stores all the dot products of every combination
    dot_prods = batch @ batch.T  # shape (B,B)

    # Sum all similarity, but overcounting 2 + a_i.a_{i+1},
    # we want 1 - a_i.a_{i+1}, so add -2*a_i.a_{i+1} - 1
    loss = torch.sum(dot_prods)
    for i in range(0, len(batch), 2):
        loss += -2 * dot_prods[i][i + 1] - 1

    return loss
