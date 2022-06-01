import torch


class SparsityLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, attn_mask, idx):
        """
        :param x: shape (2B, L, C) or (B, L, C)
        :param attn_mask: (2B, L) of which tokens in the sequence to ignore
        :param idx: shape (2B, I) of which indices of the activation to compute sparsity
        :return: Loss value
        """
        raise NotImplementedError("Abstract class, please implement.")


class L1Loss(SparsityLoss):
    def __init__(self):
        super().__init__()

    def forward(self, x, attn_mask, idx):
        """
        Computes the L1 regularisation of the activation
        :param x: shape (2B, L, C) or (B, L, C)
        :param attn_mask: (2B, L) of which tokens in the sequence to ignore
        :param idx: shape (2B, I) of which indices of the activation to compute l1
        :return: L1 loss
        """
        b, L, c = x.shape
        if idx[0, 0] != -1:
            # WARNING: this might not work on decoder layers
            _, I = idx.shape
            idx = idx.unsqueeze(2).expand(-1, -1, c)  # (2B, I, C)
            x = torch.gather(x, dim=1, index=idx)  # (2B, I, C)
        else:
            x = x*attn_mask.unsqueeze(-1).type(x.dtype)

        return torch.norm(x, 1) / x.shape[1]


class HoyerLoss(SparsityLoss):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, x, attn_mask, idx):
        b, L, c = x.shape
        if idx[0, 0] != -1:
            # WARNING: this might not work on decoder layers
            _, I = idx.shape
            idx = idx.unsqueeze(2).expand(-1, -1, c)  # (2B, I, C)
            x = torch.gather(x, dim=1, index=idx)  # (2B, I, C)
        else:
            x = x*attn_mask.unsqueeze(-1).type(x.dtype)

        return (torch.norm(x, 1).pow(2)) / (x.pow(2).sum() + self.eps)


def build_sparsity_loss(sparsity_type, **kwargs) -> SparsityLoss:
    if sparsity_type == 'l1':
        sparsity_loss = L1Loss(**kwargs)
    elif sparsity_type == 'hoyer':
        sparsity_loss = HoyerLoss(**kwargs)
    else:
        raise NotImplementedError("Invalid loss type")

    return sparsity_loss
