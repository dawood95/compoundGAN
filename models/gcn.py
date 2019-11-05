import dgl
from dgl import function as fn
from dgl.nn import pytorch as dgl_nn

import torch
from torch import nn
from torch.nn import functional as F

class MolConv(nn.Module):
    '''
    node+edge -> node
    Bond features are only incorporated using concat with node features.
    Bond features don't really evolve. Maybe they should.
    '''
    def __init__(self,
                 node_in_feats, edge_in_feats, out_feats,
                 bias=True):

        super().__init__()

        in_feats = node_in_feats + edge_in_feats
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)
        self.act = nn.SELU(True)

        # self.bn = nn.BatchNorm1d(out_feats)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        nn.init.xavier_normal_(self.weight)
        #nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def message(self, edges):
        feats = torch.cat([edges.data['feats'], edges.src['h']], -1)
        return {'fused_feats': feats}

    def forward(self, g, feat):
        norm = torch.pow(g.in_degrees().float().clamp(min=1), -0.5)
        shp = norm.shape + (1,) * (feat.dim() - 1)
        norm = torch.reshape(norm, shp).to(feat.device)

        ##feat = feat * norm
        g.ndata['h'] = feat
        g.update_all(self.message, fn.sum('fused_feats', 'sum_f'))
        feat = g.ndata['sum_f']

        feat = feat * norm
        feat = torch.matmul(feat, self.weight)
        feat = feat * norm

        if self.bias is not None:
            feat = feat + self.bias

        # feat = self.bn(feat)
        feat = self.act(feat)
        return feat


class GCN(nn.Module):
    '''
    N-layer GConv Network to convert graph to embedding
    '''
    def __init__(self, node_in_feats, edge_in_feats, out_feats,
                 hidden_feats=[64, 128, 128], bias=True):
        super().__init__()

        self.layers = nn.ModuleList([])
        in_feats = node_in_feats
        for feats in hidden_feats:
            layer = MolConv(in_feats, edge_in_feats, feats, bias=bias)
            self.layers.append(layer)
            in_feats = feats

        # Conv might not neccessarily cover entire graph.
        # Look at all embbedding to form a 'compound' embedding
        #self.pool = dgl_nn.glob.AvgPooling()#
        self.pool = dgl_nn.glob.Set2Set(feats, 4, 4)
        feats = feats * 2
        self.fc   = nn.Linear(feats, out_feats, bias=bias)

    def forward(self, G):
        feat = G.ndata['feats']
        for l in self.layers:
            feat = l(G, feat)
        feat = self.pool(G, feat)
        feat = self.fc(feat)
        return feat
