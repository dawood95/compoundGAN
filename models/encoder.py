import torch
from torch import nn
from torch.nn import functional as F

from .gcn import GCN

class Encoder(nn.Module):

    def __init__(self, node_feats, edge_feats, latent_feats, bias=True):
        super().__init__()

        self.gcn = GCN(node_feats, edge_feats, latent_feats*2,
                       [128, 128, 128, 256, latent_feats], bias)

        self.mu_fc     = nn.Linear(latent_feats * 2, latent_feats, bias)
        self.logvar_fc = nn.Linear(latent_feats * 2, latent_feats, bias)

    def forward(self, G):
        feat     = self.gcn(G)
        mu_x     = self.mu_fc(feat)
        logvar_x = self.logvar_fc(feat)
        return mu_x, logvar_x
