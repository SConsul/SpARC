import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


class Moco(nn.Module):
    def __init__(self, model, layer_names, source_len=64, tgt_len=8,
                 dim=1024, K=4096, m=0.999, T=0.07):
        super().__init__()
        
        self.K = K
        self.m = m
        self.T = T

        # Set of intermediate layer names from which activations are gotten
        self.layer_names = [l.replace('.', '_') for l in layer_names]

        self.model_q = model
        self.model_k = deepcopy(model)
        # self.model_k.to(device)

        for param_k in self.model_k.parameters():
            param_k.requires_grad = False  # not update by gradient

        # TODO: Channel issue here, needs to be dim*L
        # Buffers are registered in state dict but no grad. Are saved as attributes of module
        for l in self.layer_names:
            len_mult = source_len if 'encoder' in l else tgt_len
            attr_name = f"queue_{l}"
            self.register_buffer(attr_name, F.normalize(torch.randn(len_mult*dim, K), dim=0))
            attr_name = f"queue_ptr_{l}"
            self.register_buffer(attr_name, torch.zeros(1, dtype=torch.long))
        pass

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.model_q.parameters(), self.model_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

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

    def forward(self, activation_q, activation_k, idx, layer_name):
        b, L, c = activation_q.shape  # (2B, L, C)
        if idx[0, 0] != -1:
            _, I = idx.shape
            idx = idx.unsqueeze(2).expand(b, I, c)  # (2B, I, C)
            activation_q = torch.gather(activation_q, dim=1, index=idx)  # (2B, I, C)
            activation_k = torch.gather(activation_k, dim=1, index=idx)  # (2B, I, C)

        # Unit norm vectors across channel dim
        activation_q = F.normalize(activation_q, dim=-1)  # (2B, I, C)
        activation_k = F.normalize(activation_k, dim=-1)  # (2B, I, C)
        activation_q = activation_q.view(b, -1)  # (2B, I*C)
        activation_k = activation_k.view(b, -1)  # (2B, I*C)

        # Randomly pick one of the two sentences in each pair to form positives
        rand_pos = torch.randint(0, 2, size=(b//2,), dtype=bool)  # (B,)
        pos_mask = torch.zeros(b, dtype=bool)
        pos_mask[torch.nonzero(rand_pos).squeeze(1)*2 + 1] = True
        pos_mask[torch.nonzero(~rand_pos).squeeze(1) * 2] = True
        pos_mask.to(activation_q.device)

        q_pos = activation_q[pos_mask]  # (B, I*C)
        k_pos = activation_k[~pos_mask]  # (B, I*C)
        l_pos = torch.einsum('nc,nc->n', [q_pos, k_pos]).unsqueeze(-1)  # (B, 1)

        queue = getattr(self, f"queue_{layer_name.replace('.', '_')}")  # (I*C, K)
        l_neg = torch.einsum('nc,ck->nk', [q_pos, queue.clone().detach()])  # (B, K)

        self._dequeue_and_enqueue(k_pos)

        # logits: Bx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)  # (B, K + 1)
        logits /= self.T  # apply temperature

        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        loss = F.cross_entropy(logits, labels)
        return loss
