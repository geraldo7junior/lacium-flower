import torch
import torch.nn.functional as F

def contrastive_loss(z_i, z_j, temperature=0.1):
    sim = F.cosine_similarity(z_i, z_j)
    return -torch.log(torch.exp(sim / temperature) / torch.exp(sim / temperature).sum())

def multitask_loss(outputs, targets):
    loss = 0
    for output, target in zip(outputs, targets):
        loss += F.cross_entropy(output, target)
    return loss
