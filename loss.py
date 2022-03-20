import torch
from torch.nn import functional as F
import numpy as np
import pdb

def schedule_KL_annealing(start, stop, n_epochs, n_cycle=4, ratio=0.5):
    """
    Custom function for multiple annealing scheduling: Monotonic and cyclical_annealing
    Given number of epochs, it returns the value of the KL weight at each epoch as a list.

    Based on from: https://github.com/haofuml/cyclical_annealing/blob/master/plot/plot_schedules.ipynb
    """

    weights = np.ones(n_epochs)
    period = n_epochs/n_cycle
    step = (stop-start)/(period*ratio) # linear schedule

    for c in range(n_cycle):
        v , i = start , 0
        while v <= stop and (int(i+c*period) < n_epochs):
            weights[int(i+c*period)] = v
            v += step
            i += 1

    return weights

def loss_function(recon_x, x, mu, logvar, kl_weight):
    """
    Computes binary cross entropy and analytical expression of KL divergence used to train Variational Autoencoders

    Losses are calculated per batch (recon vs original). Their sizes are torch.Size([128, 3, 21, 21, 21])

    Total loss is reconstruction + KL divergence summed over batch
    """
    
    # reconstruction loss (MSE/BCE for image-like data)
    # CE = torch.nn.CrossEntropyLoss()(recon_x, x)
    # MSE = torch.nn.MSELoss(reduction='mean')(recon_x, x)
    MSE = 0.1 * torch.nn.MSELoss(reduction='sum')(recon_x, x)
    # BCE = F.binary_cross_entropy(recon_x, x, reduction="mean") # only takes data in range [0, 1]
    # BCEL = torch.nn.BCEWithLogitsLoss(reduction="mean")(recon_x, x)

    # KL divergence loss (with annealing)
    # KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()) # sum or mean
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # KLD = KLD * kl_weight

    return MSE + KLD, MSE, KLD
