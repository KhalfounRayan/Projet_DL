import numpy as np
import torch
import math

def compute_disentanglement(zs, ys, L=1000, M=20000):
    '''
    Metric introduced in Kim and Mnih (2018)
    '''
    N, D = zs.size()
    _, K = ys.size()
    zs_std = torch.std(zs, dim=0)
    ys_uniq = [c.unique() for c in ys.split(1, dim=1)]  # global: move out
    V = torch.zeros(D, K, device=zs_std.device)
    # ks = np.random.randint(0, K, M)      # sample fixed-factor idxs ahead of time
    ks = torch.randint(0, K, (M,), device=zs_std.device)  # Pre-generate factor indices

    # for m in range(M):
    for k in ks:
        # k = ks[m]
        fk_vals = ys_uniq[k]
        # fix fk
        # fk = fk_vals[np.random.choice(len(fk_vals))]
        fk = fk_vals[torch.randint(len(fk_vals), (1,)).item()]
        # choose L random zs that have this fk at factor k
        mask = (ys[:, k] == fk).nonzero(as_tuple=True)[0]
        # zsh = zs[ys[:, k] == fk]
        # zsh = zsh[torch.randperm(zsh.size(0))][:L]
        zsh = zs[mask][torch.randperm(mask.size(0))[:L]]
        d_star = torch.argmin(torch.var(zsh / zs_std, dim=0))
        V[d_star, k] += 1

    # return torch.max(V, dim=1)[0].sum() / M
    return V.max(dim=1)[0].sum() / M


def preprocessed_disentanglement(latents, factors, kls, threshold):
    used_mask = kls > threshold      # threshold
    latents = latents[:, used_mask]  # assumes latents is 2D
    # dropdims = (~used_mask).nonzero()
    dropdims = (~used_mask).nonzero(as_tuple=True)[0]  # Identify dropped dimensions
    # print('Removing {} latent dimensions: {}'.format(len(dropdims), list(dropdims.view(-1).cpu().numpy())))
    print('Removing {} latent dimensions: {}'.format(len(dropdims), dropdims.cpu().numpy().tolist()))
    
    return compute_disentanglement(latents, factors)


def compute_sparsity(zs, norm):
    '''
    Hoyer metric
    norm: normalise input along dimension to avoid that dimension collapse leads to good sparsity
    '''
    latent_dim = zs.size(-1)
    
    if norm:
        zs_std = zs.std(dim=0, keepdim=True)
        zs_std = zs_std.where(zs_std != 0, torch.ones_like(zs_std))  # Avoid division by zero
        zs = zs / zs_std
    
    l1_l2 = (zs.abs().sum(dim=-1) / zs.pow(2).sum(dim=-1).sqrt()).mean()
    sparsity = (math.sqrt(latent_dim) - l1_l2) / (math.sqrt(latent_dim) - 1)
    
    return sparsity
