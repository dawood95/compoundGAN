import torch
import dgl

from torch import nn
from torch.nn import functional as F

from .gcn import MolConv, GCN

class Discriminator(nn.Module):

    def __init__(self, node_feats, edge_feats, bias=True):
        super().__init__()
        hidden_feats = [128, 128, 128, 128, 128]

        self.classifiers = nn.ModuleList([])
        self.conv_layers = nn.ModuleList([])
        self.pool_layers = nn.ModuleList([])
        # self.pool = dgl.nn.pytorch.glob.AvgPooling()

        in_feats = node_feats
        self.conv_zero = MolConv(in_feats, edge_feats, 128, bias=bias)
        in_feats = 128
        for feats in hidden_feats:
            layer = MolConv(in_feats, edge_feats, feats, bias=bias)
            self.conv_layers.append(layer)

            self.pool_layers.append(dgl.nn.pytorch.glob.Set2Set(feats, 2, 2))

            classifier = nn.Linear(feats*2, 1, bias=bias)
            self.classifiers.append(classifier)

            in_feats = feats

    def forward(self, G):#, return_feat=False):

        score = 0
        feat = G.ndata['feats']

        inp_feat = self.conv_zero(G, feat)

        for i in range(len(self.classifiers)):
            feat = self.conv_layers[i](G, inp_feat)
            feat = feat + inp_feat
            pooled_feat = self.pool_layers[i](G, feat)
            score += self.classifiers[i](pooled_feat)
            inp_feat = feat

        return score

class DiscriminatorAlt(nn.Module):

    def __init__(self, node_feats, edge_feats, bias=True):
        super().__init__()
        hidden_feats = [64, 128, 128, 256, 256, 256]
        self.gcn = GCN(node_feats, edge_feats, 1, hidden_feats, bias)
        # self.classifier = nn.Linear(256, 1, bias=bias)
        # self.act = nn.SELU()#PReLU(128)

        # nn.init.xavier_normal_(self.classifier.weight)
        # if self.classifier.bias is not None:
        #     nn.init.zeros_(self.classifier.bias)

    def forward(self, G, return_feat=False):
        feat = self.gcn(G)
        # feat = self.act(feat)
        # feat = self.classifier(feat)
        return feat
