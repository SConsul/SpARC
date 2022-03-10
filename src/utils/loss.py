import torch
import torch.nn.functional as F
import numpy as np


def binary_sim_loss(batch, idx, sim_type=None):
    """
    Computes the similarity/difference loss:
    ∑[i = 0, 2, ..., 2B] (1 - a_i.a_i) + ∑[i=0,1,...]∑[j!=i,j!=i+1] a_i . a_j
    First sum is -ve cosine similarity of similar questions,
    second sum is cosine similarity of each question with every other dissimilar question.
    :param batch: shape (2B, L, C) of (a_0, a_0', a_1, a_1', ..., a_B, a_B')
        where consecutive questions are similar
    :param idx: shape (2B, I) of which indices of the activation to compute cosine sim
    :return: Loss value
    """
    b, L, c = batch.shape
    if idx[0, 0] != -1:
        _, I = idx.shape
        idx = idx.unsqueeze(2).expand(b, I, c)  # (2B, I, C)
        batch = torch.gather(batch, dim=1, index=idx)  # (2B, I, C)

    # Unit norm vectors across channel dim
    batch = F.normalize(batch, dim=-1)  # (2B, I, C)
    batch = batch.view(b, -1)  # shape = (2B, I*C)

    # batch = F.normalize(batch, dim=1)  # shape (2B, L*C)

    # stores all the dot products of every combination
    sim_matrix = batch @ batch.T  # shape (B,B)

    if sim_type == "angle":
        eps = 1e-7
        sim_matrix = torch.clamp(sim_matrix,-1+eps,1-eps)
        ang = torch.acos(sim_matrix)
        loss = -torch.sum(ang)
        for i in range(0, len(batch), 2):
            loss += 2 * ang[i][i + 1]
    else:
        # # Sum all similarity, but overcounting 2 + a_i.a_{i+1},
        # # we want 1 - a_i.a_{i+1}, so add -2*a_i.a_{i+1} - 1
        # loss = torch.sum(sim_matrix)
        # for i in range(0, len(batch), 2):
        #     loss += -2 * sim_matrix[i][i + 1] - 1

        pos_pair_mask = torch.arange(b // 2).repeat_interleave(2)  # (2B
        pos_pair_mask = (pos_pair_mask.unsqueeze(0) == pos_pair_mask.unsqueeze(1)).float()  # (2B, 2B)
        pos_pair_mask.to(batch.device)

        # Discard the main diagonal
        diag_mask = torch.eye(pos_pair_mask.shape[0], dtype=torch.bool).to(batch.device)  # (2B, 2B)
        pos_pair_mask = pos_pair_mask[~diag_mask].view(pos_pair_mask.shape[0], -1)  # (2B, 2B-1)
        sim_matrix = sim_matrix[~diag_mask].view(sim_matrix.shape[0], -1)  # (2B, 2B-1)

        # positives
        pos = sim_matrix[pos_pair_mask.bool()].view(sim_matrix.shape[0], -1)  # (2B, 1)
        neg = sim_matrix[~pos_pair_mask.bool()].view(sim_matrix.shape[0], -1)  # (2B, 2B-2)
        logits = torch.cat((pos, neg), dim=1)  # (2B, 2B-1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=batch.device)  # (2B,)
        loss = F.cross_entropy(logits, labels)

    return loss
