import torch
from torch import nn
from torch.nn import functional as F

from .gcn import GCN

class Encoder(nn.Module):

    def __init__(self, node_feats, edge_feats, latent_feats, bias=True):
        super().__init__()

        self.gcn = GCN(node_feats, edge_feats, 256, [128, 128, 128, 256], bias)

        self.mu_fc     = nn.Linear(256, latent_feats, bias)
        self.logvar_fc = nn.Linear(256, latent_feats, bias)

    def forward(self, G):
        feat     = F.relu(self.gcn(G))
        mu_x     = self.mu_fc(feat)
        logvar_x = self.logvar_fc(feat)
        return mu_x, logvar_x

    def reparameterize(self, mu, logvar, no_noise=False):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        if not no_noise:
            return mu + std*eps
        else:
            return mu
