import dgl
import torch
import numpy as np

from torch import nn
from torch.nn import functional as F
from torch.distributions import RelaxedOneHotCategorical as gumbel

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

        node_input_size = hidden_size
        # node_input_size = sum(node_feats)
        edge_input_size = hidden_size

        self.latent_project = nn.ModuleList()
        for i in range(num_layers):
            self.latent_project.append(nn.Sequential(
                nn.Linear(latent_size, hidden_size, bias=True),
                nn.SELU(True), #PReLU(hidden_size),
            ))

        self.node_cell = DecoderCell(node_input_size, hidden_size, num_layers, bias)
        self.edge_cell = DecoderCell(edge_input_size, hidden_size, num_layers, bias)

        self.node_classifiers = nn.ModuleList()
        self.edge_classifiers = nn.ModuleList()

        for feat_size in node_feats:
            self.node_classifiers.append(nn.Linear(hidden_size, feat_size, bias))
        for feat_size in edge_feats:
            self.edge_classifiers.append(nn.Linear(hidden_size, feat_size, bias))

        self.end_node      = node_feats[0]-1
        self.node_inp_size = node_input_size

        self.reset_parameters()

    def reset_parameters(self):
        return

    def forward(self, z, max_nodes=50, gumbel_temp=0.001):

        batch_size = z.shape[0]

        # reset context
        self.node_cell.reset_hidden(batch_size)
        self.edge_cell.reset_hidden(batch_size)

        num_nodes  = -1*torch.ones(batch_size)
        num_nodes  = num_nodes.cuda() if z.is_cuda else num_nodes

        ctx = []
        for l in self.latent_project:
            x = l(z)
            ctx.append(x.unsqueeze(0))
            # ctx.append(z.unsqueeze(0))
        ctx = torch.cat(ctx, 0)

        # Set context
        self.node_cell.hidden = ctx

        pred_node_feats = []
        pred_node_logprobs = []

        pred_edge_feats = []
        pred_edge_logprobs = []

        node_embeddings = []

        for i in range(max_nodes):

            # Create input for nodeRNN
            if i == 0:
                x = torch.zeros((batch_size, self.node_inp_size))
                if z.is_cuda: x = x.cuda()
            else:
                x = node_embeddings[-1]
                # x = pred_node_feats[-1].squeeze(1)

            # Generate a node
            node_emb  = self.node_cell.forward_unit(x)

            node_pred = [c(node_emb) for c in self.node_classifiers]

            node_logprobs = [F.log_softmax(p, -1) for p in node_pred]
            node_pred     = [gumbel(gumbel_temp, logits=p).rsample() for p in node_pred]
            for j, p in enumerate(node_pred):
                node_logprobs[j] = torch.sum((p.detach() * node_logprobs[j]), -1).unsqueeze(-1)

            cond1 = node_pred[0].argmax(1) == self.end_node
            cond2 = num_nodes == -1
            num_nodes[cond1 & cond2] = i+1

            # Save node features
            node_pred = torch.cat(node_pred, -1)
            node_logprobs = torch.cat(node_logprobs, -1)

            pred_node_feats.append(node_pred.unsqueeze(1))
            pred_node_logprobs.append(node_logprobs.unsqueeze(1))

            # Nothing to run for first node
            # NOTE: self-loop feature shouldn't be differentiable
            # So, it's not predicted
            if i == 0:
                node_embeddings.append(node_emb)
                continue

            node_state = self.node_cell.hidden
            # self.edge_cell.reset_hidden(batch_size)
            self.edge_cell.hidden += self.node_cell.hidden

            edge_inp = [prev_nemb.unsqueeze(0) for prev_nemb in node_embeddings[::-1][:12]]
            edge_inp = torch.cat(edge_inp, 0)

            edge_emb_seq = self.edge_cell.forward_seq(edge_inp)
            edge_emb_seq = edge_emb_seq[list(range(len(edge_emb_seq)))[::-1]]
            edge_emb_seq = edge_emb_seq.view(-1, edge_emb_seq.shape[-1])

            edge_pred_seq = [c(edge_emb_seq) for c in self.edge_classifiers]
            # if gumbel_temp is None:
            #    edge_pred_seq = [F.softmax(p, -1) for p in edge_pred_seq]
            # else:
            edge_pred_seq = [gumbel(gumbel_temp, logits=p).rsample() for p in edge_pred_seq]
            edge_logprobs_seq = [F.log_softmax(p, -1) for p in edge_pred_seq]
            for j, p in enumerate(edge_pred_seq):
                edge_logprobs_seq[j] = torch.sum((p.detach() * edge_logprobs_seq[j]), -1).unsqueeze(-1)

            edge_types = edge_pred_seq[0].argmax(-1)

            edge_pred_seq = torch.cat(edge_pred_seq, -1)
            edge_logprobs_seq = torch.cat(edge_logprobs_seq, -1)

            edge_pred_seq = edge_pred_seq.view(len(edge_inp), batch_size, -1)
            edge_logprobs_seq = edge_logprobs_seq.view(len(edge_inp), batch_size, -1)
            edge_types = edge_types.view(len(edge_inp), batch_size)
            edge_types = edge_types.sum(0)

            pred_edge_feats.append(edge_pred_seq)
            pred_edge_logprobs.append(edge_logprobs_seq)

            # Reset nodeRNN hidden state to output from edgeRNN
            # self.node_cell.reset_hidden(batch_size)
            self.node_cell.hidden = node_state + self.edge_cell.hidden
            node_embeddings.append(node_emb)

            # num_nodes[(edge_types == 0) & (num_nodes == -1)] = i + 1
            if (num_nodes != -1).all(): break

            # if (i > 12 and (i - 12)%2==0) or (((i * (i-1))/2) % 24 == 0):
            #     self.node_cell.hidden.detach_()
            #     node_embeddings = [n.detach_() for n in node_embeddings]

        pred_node_feats = torch.cat(pred_node_feats, 1)
        pred_node_logprobs = torch.cat(pred_node_logprobs, 1)

        if len(pred_edge_feats) > 0:
            pred_edge_feats = torch.cat(pred_edge_feats, 0)
            pred_edge_logprobs = torch.cat(pred_edge_logprobs, 0)

        num_nodes[num_nodes == -1] = pred_node_feats.shape[1]

        return num_nodes, pred_node_feats, pred_edge_feats, pred_node_logprobs, pred_edge_logprobs

    def create_graph(self, num_nodes, pred_node_feats, pred_edge_feats,
                     pred_node_logprobs, pred_edge_logprobs):

        batch_size = pred_node_feats.shape[0]

        output_graphs = [dgl.DGLGraph() for _ in range(batch_size)]
        for b in range(batch_size):
            num_node = int(num_nodes[b].item())
            output_graphs[b].add_nodes(num_node)
            output_graphs[b].ndata['feats'] = pred_node_feats[b, :num_node]
            output_graphs[b].ndata['logprobs'] = pred_node_logprobs[b, :num_node].sum(-1)

            if len(pred_edge_feats) == 0:
                raise Exception

            if num_node > 12:
                num_edge = (12*11) + (num_node - 12)*24
            else:
                num_edge = (num_node * (num_node - 1)) # + for self loop
            num_prede = num_edge//2

            edge_feats = torch.zeros((num_edge + num_node, pred_edge_feats.shape[-1]))
            edge_logprobs = torch.zeros((num_edge + num_node,))
            edge_isgen = torch.zeros((num_edge + num_node,))

            edge_isgen[:num_prede] = 1

            edge_feats[:num_prede] = pred_edge_feats[:num_prede, b, :]
            edge_feats[num_prede:2*num_prede] = pred_edge_feats[:num_prede, b, :]

            edge_logprobs[:num_prede] = pred_edge_logprobs[:num_prede, b].sum(-1)
            edge_logprobs[num_prede:2*num_prede] = pred_edge_logprobs[:num_prede, b].sum(-1)

            edge_start = []
            edge_end   = []
            for i in range(num_node):
                for j in range(i):
                    if ((i > 12) and (j < (i - 12))): continue
                    edge_start.append(i)
                    edge_end.append(j)

            edge_isgen[2*num_prede:] = -1

            output_graphs[b].add_edges(edge_start, edge_end)
            output_graphs[b].add_edges(edge_end, edge_start)
            output_graphs[b].add_edges(list(range(num_node)), list(range(num_node)))
            output_graphs[b].edata['feats'] = edge_feats
            output_graphs[b].edata['logprobs'] = edge_logprobs
            output_graphs[b].edata['isgen'] = edge_isgen

        G = dgl.batch(output_graphs)
        G.to(pred_node_feats.device)
        return G
