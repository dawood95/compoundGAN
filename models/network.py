import torch
import numpy as np
from torch import nn
from math import pi, log

from .encoder import Encoder
from .decoder import Decoder

from .cnf.ode import ODEnet, ODEfunc
from .cnf.cnf import CNF
from .cnf.flow import SequentialFlow
from .cnf.normalization import MovingBatchNorm1d


class CVAEF(nn.Module):
    '''
    Compound VAE Flow
    '''
    def __init__(self, node_dims, edge_dims, latent_dim=256,
                 cnf_hidden_dims=[256,], cnf_context_dim=0,
                 cnf_T=1.0, cnf_train_T=False,
                 solver='dopri5', atol=1e-5, rtol=1e-5, use_adjoint=True,
                 num_decoder_layers=4):
        super().__init__()

        self.encoder = Encoder(sum(node_dims), sum(edge_dims), latent_dim)
        self.decoder = Decoder(latent_dim, node_dims, edge_dims,
                               num_decoder_layers)

        diffeq   = ODEnet(latent_dim, cnf_hidden_dims, cnf_context_dim)
        odefunc  = ODEfunc(diffeq)
        self.cnf = CNF(odefunc, cnf_T, cnf_train_T,
                       solver, atol, rtol, use_adjoint)

    def forward(self, x):
        raise NotImplementedError

    @staticmethod
    def gaussian_entropy(logvar):
        const = 0.5 * float(logvar.size(-1)) * (1. + np.log(np.pi * 2))
        ent = 0.5 * logvar.sum(dim=-1, keepdim=False) + const
        return ent

    @staticmethod
    def stdnormal_logprob(z):
        log_z = -0.5 * z.size(-1) * log(2 * pi)
        return log_z - z.pow(2) / 2

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std*eps

    def calc_loss(self, G, atom_x, atom_y, bond_y, seq_len):

        mu, logvar = self.encoder(G)
        z = self.reparameterize(mu, logvar)

        reconstruction_loss = self.decoder.calc_loss(z, atom_x, atom_y, bond_y, seq_len)

        # return reconstruction_loss, torch.Tensor([0.,]), torch.Tensor([0.,])

        entropy = self.gaussian_entropy(logvar).mean()
        entropy_loss = -entropy

        # return reconstruction_loss, entropy_loss, torch.Tensor([0.,])

        w, delta_log_pw = self.cnf(z, None)
        log_pw = self.stdnormal_logprob(w).sum(-1, keepdim=True)
        log_pz = log_pw - delta_log_pw
        prior_loss = -log_pz.mean()

        # return torch.Tensor([0.,]), entropy_loss, prior_loss

        return reconstruction_loss, entropy_loss, prior_loss
