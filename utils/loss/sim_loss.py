import torch
import torch.nn.functional as F


class SimLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, attn_mask, idx, layer_name):
        """
        :param x: shape (2B, L, C) of (a_0, a_0', a_1, a_1', ..., a_B, a_B')
            where consecutive questions are similar
        :param attn_mask: (2B, L) of which tokens in the sequence to ignore
        :param idx: shape (2B, I) of which indices of the activation to compute cosine sim
        :param layer_name: Layer name of activation on which this loss is being computed
        :return: Loss value
        """
        raise NotImplementedError("Abstract class, please implement.")


class BatchSimLoss(SimLoss):
    """
    Computes the similarity/difference loss:
    ∑[i = 0, 2, ..., 2B] (1 - a_i.a_i) + ∑[i=0,1,...]∑[j!=i,j!=i+1] a_i . a_j
    First sum is -ve cosine similarity of similar questions,
    second sum is cosine similarity of each question with every other dissimilar question.
    """
    def __init__(self):
        super().__init__()

    def get_sim_matrix(self, x, attn_mask, idx):
        b, L, c = x.shape
        if idx[0, 0] != -1:
            _, I = idx.shape
            idx = idx.unsqueeze(2).expand(-1, -1, c)  # (2B, I, C)
            x = torch.gather(x, dim=1, index=idx)  # (2B, I, C)

            # Unit norm vectors across channel dim, avg across num tokens in seq dim
            x = F.normalize(x, dim=-1)  # (2B, I, C)
            x = x.view(b, -1) / x.shape[1]  # shape = (2B, I*C)
        else:
            num_nonzero = attn_mask.sum(-1).view(-1, 1).type(x.dtype)  # (2B, 1)
            mask = attn_mask.unsqueeze(-1).type(x.dtype)  # (2B, L, 1)

            # Average the token vector across the sequence (non-padding tokens)
            x = (x * mask).sum(1)/num_nonzero  # (2B, C)

        # batch = F.normalize(x, dim=1)  # shape (2B, L*C)

        # stores all the dot products of every combination
        sim_matrix = x @ x.T  # shape (2B,2B)
        return sim_matrix

    def forward(self, x, attn_mask, idx, layer_name):
        b, L, c = x.shape
        sim_matrix = self.get_sim_matrix(x, attn_mask, idx)

        # # Sum all similarity, but overcounting 2 + a_i.a_{i+1},
        # # we want 1 - a_i.a_{i+1}, so add -2*a_i.a_{i+1} - 1
        # loss = torch.sum(sim_matrix)
        # for i in range(0, len(batch), 2):
        #     loss += -2 * sim_matrix[i][i + 1] - 1

        pos_pair_mask = torch.arange(b // 2).repeat_interleave(2)  # (2B,)
        pos_pair_mask = (pos_pair_mask.unsqueeze(0) == pos_pair_mask.unsqueeze(1)).float()  # (2B, 2B)
        pos_pair_mask.to(x.device)

        # Discard the main diagonal
        diag_mask = torch.eye(pos_pair_mask.shape[0], dtype=torch.bool).to(x.device)  # (2B, 2B)
        pos_pair_mask = pos_pair_mask[~diag_mask].view(pos_pair_mask.shape[0], -1)  # (2B, 2B-1)
        sim_matrix = sim_matrix[~diag_mask].view(sim_matrix.shape[0], -1)  # (2B, 2B-1)

        # positives
        pos = sim_matrix[pos_pair_mask.bool()].view(sim_matrix.shape[0], -1)  # (2B, 1)
        neg = sim_matrix[~pos_pair_mask.bool()].view(sim_matrix.shape[0], -1)  # (2B, 2B-2)
        logits = torch.cat((pos, neg), dim=1)  # (2B, 2B-1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=x.device)  # (2B,)
        loss = F.cross_entropy(logits, labels)

        return loss


class BatchAngleLoss(BatchSimLoss):
    def __init__(self):
        super().__init__()

    def forward(self, x, attn_mask, idx, layer_name):
        sim_matrix = self.get_sim_matrix(x, attn_mask, idx)

        eps = 1e-7
        sim_matrix = torch.clamp(sim_matrix, -1 + eps, 1 - eps)
        ang = torch.acos(sim_matrix)
        loss = -torch.sum(ang)
        for i in range(0, len(x), 2):
            loss += 2 * ang[i][i + 1]

        return loss


class Moco(SimLoss):

    def __init__(self, layer_names, source_len=64, tgt_len=8,
                 dim=1024, K=4096, m=0.999, T=0.07):
        super().__init__()

        self.K = K
        self.m = m
        self.T = T

        # Set of intermediate layer names from which activations are gotten
        self.layer_names = [l.replace('.', '_') for l in layer_names]

        # TODO: Channel issue here, needs to be dim*L
        # Buffers are registered in state dict but no grad. Are saved as attributes of module
        for l in self.layer_names:
            len_mult = source_len if 'encoder' in l else tgt_len
            attr_name = f"queue_{l}"
            self.register_buffer(attr_name, F.normalize(torch.randn(len_mult * dim, K), dim=0))
            attr_name = f"queue_ptr_{l}"
            self.register_buffer(attr_name, torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, layer_name, keys):
        batch_size = keys.shape[0]

        queue = getattr(self, f"queue_{layer_name.replace('.', '_')}")
        q_ptr = getattr(self, f"queue_ptr_{layer_name.replace('.', '_')}")
        ptr = int(q_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        q_ptr[0] = ptr

    def forward(self, activation, attn_mask, idx, layer_name):
        b, L, c = activation.shape  # (2B, L, C)
        if idx[0, 0] != -1:
            _, I = idx.shape
            idx = idx.unsqueeze(2).expand(b, I, c)  # (2B, I, C)
            activation = torch.gather(activation, dim=1, index=idx)  # (2B, I, C)

        # Unit norm vectors across channel dim, avg across num tokens in seq dim
        activation = F.normalize(activation, dim=-1)  # (2B, I, C)
        activation = activation.view(b, -1) / activation.shape[1]  # (2B, I*C)

        # Randomly pick one of the two sentences in each pair to form positives
        rand_pos = torch.randint(0, 2, size=(b // 2,), dtype=bool)  # (B,)
        pos_mask = torch.zeros(b, dtype=bool)
        pos_mask[torch.nonzero(rand_pos).squeeze(1) * 2 + 1] = True
        pos_mask[torch.nonzero(~rand_pos).squeeze(1) * 2] = True
        pos_mask.to(activation.device)

        q_pos = activation[pos_mask]  # (B, I*C)
        k_pos = activation[~pos_mask]  # (B, I*C)
        l_pos = torch.einsum('nc,nc->n', [q_pos, k_pos]).unsqueeze(-1)  # (B, 1)

        queue = getattr(self, f"queue_{layer_name.replace('.', '_')}")  # (I*C, K)
        l_neg = torch.einsum('nc,ck->nk', [q_pos, queue.clone().detach()])  # (B, K)

        self._dequeue_and_enqueue(layer_name, k_pos)

        # logits: Bx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)  # (B, K + 1)
        logits /= self.T  # apply temperature

        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        loss = F.cross_entropy(logits, labels)
        return loss


def build_sim_loss(sim_type, layer_names, src_len, tgt_len, **kwargs) -> SimLoss:
    if sim_type == 'batch':
        sim_loss = BatchSimLoss(**kwargs)
    elif sim_type == 'angle':
        sim_loss = BatchAngleLoss(**kwargs)
    elif sim_type == 'moco':
        sim_loss = Moco(layer_names, src_len, tgt_len, **kwargs)
    else:
        raise NotImplementedError("Invalid loss type")

    return sim_loss
