import torch


def gumbel_sigmoid(logits: torch.Tensor, tau=1., hard=False, eps=1e-10) -> torch.Tensor:
    uniform = logits.new_empty([2]+list(logits.shape)).uniform_(0, 1)

    noise = -((uniform[1] + eps).log() / (uniform[0] + eps).log() + eps).log()
    res = torch.sigmoid((logits + noise) / tau)

    if hard:
        res = ((res > 0.5).type_as(res) - res).detach() + res

    return res
