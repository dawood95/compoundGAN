import torch
import numpy as np

from math import pi, log
from torch import nn
from torch.nn import functional as F

from .encoder import Encoder
from .decoder import Decoder

from .cnf.ode import ODEnet, ODEfunc
from .cnf.cnf import CNF
from .cnf.normalization import MovingBatchNorm1d


class CVAEF(nn.Module):
    '''
    Compound VAE Flow
    '''
    def __init__(self, input_dims, latent_dim=256,
                 cnf_hidden_dims=[256,], cnf_condition_dim=0,
                 cnf_T=1.0, cnf_train_T=False,
                 solver='dopri5', atol=1e-5, rtol=1e-5, use_adjoint=True,
                 num_encoder_layers=4, num_decoder_layers=4):
        super().__init__()

        self.encoder = Encoder(sum(input_dims), 128, latent_dim,
                               num_encoder_layers, num_head=4,
                               ff_dim=1024, dropout=0.1)

        self.decoder = Decoder(latent_dim, input_dims,
                               num_decoder_layers, num_head=4,
                               ff_dim=1024, dropout=0.1)

        diffeq   = ODEnet(latent_dim, cnf_hidden_dims)
        odefunc  = ODEfunc(diffeq)
        self.cnf = CNF(odefunc, cnf_T, cnf_train_T,
                       solver, atol, rtol, use_adjoint)

        self.condition_dim = cnf_condition_dim

        return
    
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

    def forward(self, data):

        emb, emb_mask, token_x, token_y, condition_y = data

        mu, logvar = self.encoder(emb, emb_mask)
        z = self.reparameterize(mu, logvar)

        reconstruction_loss = self.decoder(z, token_x, token_y)

        # return reconstruction_loss, torch.Tensor([0.,]), torch.Tensor([0.,])

        entropy = self.gaussian_entropy(logvar).mean()
        entropy_loss = -entropy

        # return reconstruction_loss, entropy_loss, torch.Tensor([0.,])

        w, delta_log_pw = self.cnf(z)
        w, condition_x = w[:, :-self.condition_dim], w[:, -self.condition_dim:]
        
        log_pw = self.stdnormal_logprob(w).sum(-1, keepdim=True)
        log_pz = log_pw - delta_log_pw
        prior_loss = -log_pz.mean() + F.mse_loss(condition_x, condition_y)*1e3
        
        # return torch.Tensor([0.,]), entropy_loss, prior_loss

        return reconstruction_loss, entropy_loss, prior_loss
