import dgl
import torch
import numpy as np

from torch import nn
from torch.nn import functional as F
import random

from .decoder_cell import DecoderCell

class Generator(nn.Module):
    '''
    z --> Graph
    '''
    def __init__(self, latent_size, node_feats, edge_feats,
                 num_layers=1, bias=True):

        super().__init__()

        hidden_size = 256
        input_size  = hidden_size

        self.latent_project = nn.ModuleList()
        for i in range(num_layers):
            self.latent_project.append(nn.Linear(latent_size, hidden_size, bias=True))

        self.node_cell = DecoderCell(input_size, hidden_size, num_layers, bias)
        self.edge_cell = DecoderCell(input_size, hidden_size, num_layers, bias)

        self.node_classifiers = nn.ModuleList()
        self.edge_classifiers = nn.ModuleList()

        for feat_size in node_feats:
            self.node_classifiers.append(nn.Linear(hidden_size, feat_size, bias))
        for feat_size in edge_feats:
            self.edge_classifiers.append(nn.Linear(hidden_size, feat_size, bias))

        self.end_node      = node_feats[0]-1
        self.node_inp_size = input_size
        self.num_layers    = num_layers

    def forward(self, z, max_nodes=50):

        batch_size = z.shape[0]

        num_nodes     = -1*torch.ones(batch_size)
        num_nodes     = num_nodes.cuda() if z.is_cuda else num_nodes

        ctx = []
        for l in self.latent_project:
            x = l(z)
            x = F.selu(x)
            ctx.append(x.unsqueeze(0))
        ctx = torch.cat(ctx, 0)

        # Set nodeRNN hidden state to ctx
        self.node_cell.reset_hidden(batch_size)
        self.node_cell.hidden = ctx

        pred_node_feats = []
        pred_edge_feats = []

        node_embeddings = []
        for i in range(max_nodes):

            # Create input for nodeRNN
            if i == 0:
                x = torch.zeros((batch_size, self.node_inp_size))
                if z.is_cuda: x = x.cuda()
            else:
                x = node_embeddings[-1]

            # Generate a node
            node_emb  = self.node_cell.forward_unit(x)
            node_pred = [c(node_emb) for c in self.node_classifiers]
            node_pred = [F.softmax(p, -1) for p in node_pred]

            cond1 = node_pred[0].argmax(1) == self.end_node
            cond2 = num_nodes == -1
            num_nodes[cond1 & cond2] = i+1

            # Save node features
            node_pred = torch.cat(node_pred, 1)
            pred_node_feats.append(node_pred.unsqueeze(1))

            # Nothing to run for first node
            # NOTE: self-loop feature shouldn't be differentiable
            # So, it's not predicted
            if i == 0:
                node_embeddings.append(node_emb)
                continue

            self.edge_cell.reset_hidden(batch_size)
            self.edge_cell.hidden = self.node_cell.hidden

            edge_inp = [prev_nemb.unsqueeze(0) for prev_nemb in node_embeddings[::-1]]
            edge_inp = torch.cat(edge_inp, 0)
            edge_emb_seq = self.edge_cell.forward_seq(edge_inp)
            edge_emb_seq = edge_emb_seq[list(range(len(edge_emb_seq)))[::-1]]
            edge_emb_seq = edge_emb_seq.view(-1, edge_emb_seq.shape[-1])

            edge_pred_seq = [c(edge_emb_seq) for c in self.edge_classifiers]
            edge_pred_seq = [F.softmax(p, -1) for p in edge_pred_seq]
            edge_pred_seq = torch.cat(edge_pred_seq, -1)
            edge_pred_seq = edge_pred_seq.view(len(edge_inp), batch_size, -1)
            pred_edge_feats.append(edge_pred_seq)

            # Reset nodeRNN hidden state to output from edgeRNN
            self.node_cell.reset_hidden(batch_size)
            self.node_cell.hidden = self.edge_cell.hidden
            node_embeddings.append(node_emb)

            if (num_nodes != -1).all(): break

        pred_node_feats = torch.cat(pred_node_feats, 1)

        if len(pred_edge_feats) > 0:
            pred_edge_feats = torch.cat(pred_edge_feats, 0)

        num_nodes[num_nodes == -1] = pred_node_feats.shape[1]

        return num_nodes, pred_node_feats, pred_edge_feats

    def create_graph(self, num_nodes, pred_node_feats, pred_edge_feats):
        batch_size = pred_node_feats.shape[0]

        output_graphs = [dgl.DGLGraph() for _ in range(batch_size)]
        for b in range(batch_size):
            num_node = int(num_nodes[b].item())
            output_graphs[b].add_nodes(num_node)
            output_graphs[b].ndata['feats'] = pred_node_feats[b, :num_node]

            num_edge = (num_node * (num_node - 1)) + num_node
            num_prede = (num_node * (num_node - 1)) // 2

            if len(pred_edge_feats) == 0: continue

            edge_feats = torch.zeros((num_edge, pred_edge_feats.shape[-1]))
            edge_feats[:num_prede] = pred_edge_feats[:num_prede, b, :]
            edge_feats[num_prede:2*num_prede] = pred_edge_feats[:num_prede, b, :]

            edge_start = [j for i in range(num_node) for j in range(i)]
            edge_end = [i for i in range(num_node) for j in range(i)]

            output_graphs[b].add_edges(edge_start, edge_end)
            output_graphs[b].add_edges(edge_end, edge_start)
            output_graphs[b].add_edges(list(range(num_node)), list(range(num_node)))
            output_graphs[b].edata['feats'] = edge_feats

        G = dgl.batch(output_graphs)
        G.to(pred_node_feats.device)
        return G
