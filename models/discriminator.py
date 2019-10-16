import torch
import dgl

from torch import nn
from torch.nn import functional as F

from .gcn import MolConv, GCN

class Discriminator(nn.Module):

    def __init__(self, node_feats, edge_feats, bias=True):
        super().__init__()
        hidden_feats = [64, 128, 128, 256]

        self.classifiers = nn.ModuleList([])
        self.conv_layers = nn.ModuleList([])
        self.pool = dgl.nn.pytorch.glob.AvgPooling()

        in_feats = node_feats
        for feats in hidden_feats:
            layer = MolConv(in_feats, edge_feats, feats, bias=bias)
            classifier = nn.Linear(feats, 1, bias=bias)
            self.conv_layers.append(layer)
            self.classifiers.append(classifier)
            in_feats = feats

    def forward(self, G):#, return_feat=False):

        score = None
        feat = G.ndata['feats']
        for i in range(len(self.classifiers)):
            feat = self.conv_layers[i](G, feat)
            if score is None:
                score = self.pool(G, self.classifiers[i](feat))#.mean()
            else:
                score += self.pool(G, self.classifiers[i](feat))#.mean()

        return score

class DiscriminatorAlt(nn.Module):

    def __init__(self, node_feats, edge_feats, bias=True):
        super().__init__()
        hidden_feats = [64, 128, 256, 256]
        self.gcn = GCN(node_feats, edge_feats, 256, hidden_feats, bias)
        self.classifier = nn.Linear(256, 1, bias=bias)
        self.act = nn.SELU(True)#PReLU(128)

        # nn.init.xavier_normal_(self.classifier.weight)
        # if self.classifier.bias is not None:
        #     nn.init.zeros_(self.classifier.bias)

    def forward(self, G, return_feat=False):
        feat = self.gcn(G)
        feat = self.act(feat)
        feat = self.classifier(feat)
        return feat
