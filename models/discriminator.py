import torch

from torch import nn
from torch.nn import functional as F

from .gcn import GCN

class Discriminator(nn.Module):

    def __init__(self, node_feats, edge_feats, bias=True):
        super().__init__()
        out_feats    = 128
        hidden_feats = [64, 128, 128, 128]
        self.gcn     = GCN(node_feats, edge_feats, out_feats, hidden_feats, bias)
        self.classifier = nn.Linear(out_feats, 1, bias)

    def forward(self, G):
        feat = F.relu(self.gcn(G))
        pred = self.classifier(feat)
        return pred
