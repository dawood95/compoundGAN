import dgl

from dgl.nn import pytorch as dgl_nn
from torch import nn

class SGConv(nn.Module):
    def __init__(self, in_feats, out_feats,
                 k=1, bias=True, norm=None):
        super(SGConv, self).__init__()
        self.fc = nn.Linear(in_feats, out_feats, bias=bias)
        self._k = k
        self.norm = norm

    def forward(self, graph):
        graph = graph.local_var()
        degs = graph.in_degrees().float().clamp(min=1)
        norm = th.pow(degs, -0.5)
        norm[th.isinf(norm)] = 0
        norm = norm.to(feat.device).unsqueeze(1)

        # compute first set of feats
        
        # compute (D^-1 A D) X
        for _ in range(self._k):
            feat = feat * norm
            graph.ndata['h'] = feat
            graph.update_all(fn.copy_u('h', 'm'),
                             fn.sum('m', 'h'))
            feat = graph.ndata.pop('h')
            feat = feat * norm

        if self.norm is not None:
            feat = self.norm(feat)

        return self.fc(feat)


class SGCN(nn.Module):
    '''
    N-layer SGConv Network to convert graph to embedding
    '''
    def __init__(self, in_feats, out_feats,
                 hidden_feats=[64, 128, 128], k=3, bias=True):
        super().__init__()

        self.layers = nn.ModuleList([])
        for feats in hidden_feats:
            layer = dgl_nn.conv.SGConv(in_feats, feats, k=k, bias=bias)
            self.layers.append(layer)
            in_feats = feats

        # Conv might not neccessarily cover entire graph.
        # Look at all embbedding to form a 'compound' embedding
        self.pool = dgl_nn.glob.Set2Set(feats, 2, 2)

    def forward(self, G):

        feats = torch.cat(G.ndata[])
