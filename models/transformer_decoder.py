import dgl
import torch
import numpy as np

from torch import nn
from torch.nn import functional as F
from torch.nn.init import xavier_uniform_

from dgl.nn.pytorch.conv import GraphConv

class Decoder(nn.Module):

    def __init__(self, latent_dim, node_feats, edge_feats, num_layers=1, bias=True,
                 num_head=1, ff_dim=1024):

        super().__init__()

        hidden_dim     = latent_dim
        node_input_dim = sum(node_feats)

        self.node_project = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim, bias=bias),
            nn.LayerNorm(hidden_dim),
            nn.SELU(True)
        )

        self.node_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(hidden_dim, num_head, ff_dim),
            num_layers,
        )

        self.gcn = nn.ModuleList([])
        for i in range(6):
            activation = nn.Sequential(nn.LayerNorm(hidden_dim), nn.SELU(True))
            self.gcn.append(
                GraphConv(hidden_dim, hidden_dim, activation=activation)
            )

        self.edge_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(hidden_dim, num_head, ff_dim),
            num_layers,
        )

        self.node_classifiers = nn.ModuleList([
            nn.Linear(hidden_dim, feat_dim, bias)
            for feat_dim in node_feats
        ])


        self.edge_classifiers = nn.ModuleList([
            nn.Linear(hidden_dim, feat_dim, bias)
            for feat_dim in edge_feats
        ])

        self.end_node       = node_feats[0] - 1
        self.node_input_dim = node_input_dim
        self.hidden_dim     = hidden_dim
        self.one_hot_dims   = node_feats

        sin_freq = torch.arange(0, hidden_dim, 2.0) / (hidden_dim)
        sin_freq = 1 / (1_000 ** sin_freq)

        cos_freq = torch.arange(1, hidden_dim, 2.0) / (hidden_dim)
        cos_freq = 1 / (1_000 ** cos_freq)

        x = torch.arange(0, 100) # 100 = max nodes

        sin_emb = torch.sin(torch.einsum('i,d->id', x, sin_freq))
        cos_emb = torch.cos(torch.einsum('i,d->id', x, cos_freq))
        sin_emb = sin_emb.unsqueeze(1)
        cos_emb = cos_emb.unsqueeze(1)

        embedding = torch.zeros(len(x), 1, hidden_dim)
        embedding[:, :, 0:self.hidden_dim:2] = sin_emb
        embedding[:, :, 1:self.hidden_dim:2] = cos_emb

        self.pos_emb = embedding

        edge_start = []
        edge_end   = []
        for i in range(len(x)):
            for j in range(i):
                if ((i > 12) and (j < (i - 12))): continue
                edge_start.append(i)
                edge_end.append(j)

        self.edge_idx = (edge_start, edge_end)

        self._reset_parameters()

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.node_decoder.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
        for p in self.edge_decoder.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def build_inputs(self, batch_size, seq_length, pos=None):

        if pos is None:
            embedding = self.pos_emb[:seq_length].clone().detach()
        else:
            embedding = self.pos_emb[pos].unsqueeze(0).clone().detach()

        embedding = embedding.repeat_interleave(batch_size, dim=1)
        return embedding

        '''
        x = torch.arange(0, seq_length)

        if pos is not None:
            x[:] = pos
            assert seq_length == 1

        sin_emb = torch.sin(torch.einsum('i,d->id', x, self.sin_freq))
        cos_emb = torch.cos(torch.einsum('i,d->id', x, self.cos_freq))

        sin_emb = sin_emb.unsqueeze(1).repeat_interleave(batch_size, dim=1)
        cos_emb = cos_emb.unsqueeze(1).repeat_interleave(batch_size, dim=1)

        embedding[:, :, 0:self.hidden_dim:2] = sin_emb.clone().detach()
        embedding[:, :, 1:self.hidden_dim:2] = cos_emb.clone().detach()

        return embedding
        '''

    def predict_edge(self, edges):
        src_feats = edges.src['emb']
        dst_feats = edges.dst['emb']
        feats = torch.cat([src_feats, dst_feats], -1)
        feats = [c(feats) for c in self.edge_classifiers]
        preds = {}
        for i in range(len(feats)):
            preds[str(i)] = feats[i]
        return preds

    def generate_square_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), 1)
        mask = mask.float().masked_fill(mask == 1, float('-inf'))
        mask = mask.masked_fill(mask == 0, float(0.0))
        return mask

    def calc_node_loss(self, node_pred, node_target):
        total_node_loss = 0
        for i in range(node_target.shape[-1]):
            loss = F.cross_entropy(node_pred[i], node_target[:, i],
                                   ignore_index=-1, reduction='mean')
            total_node_loss += loss
        return total_node_loss

    def forward(self, z, atom_i, atom_x, atom_y, bond_y):

        batch_size = z.shape[0]
        seq_length = atom_y.shape[0]

        atom_x = self.node_project(atom_x)

        pos_emb        = self.build_inputs(batch_size, seq_length).to(z)
        decoder_inputs = pos_emb + atom_x

        decoder_mask = self.generate_square_mask(seq_length).to(z)
        padding_mask = (atom_y[:seq_length, :, 0] == -1).T

        node_emb = self.node_decoder(
            tgt = decoder_inputs,
            memory = z.unsqueeze(0),
            tgt_mask = decoder_mask,
            tgt_key_padding_mask = padding_mask
        )

        node_preds  = [c(node_emb) for c in self.node_classifiers]
        node_preds  = [pred.view(-1, pred.shape[-1]) for pred in node_preds]
        node_target = atom_y.view(-1, atom_y.shape[-1])
        node_loss   = self.calc_node_loss(node_preds, node_target)

        '''
        atom_x = []
        for j, emb_size in enumerate(self.one_hot_dims):
            target = atom_y[:, :, j].data.cpu().clone()
            target[target == -1] = emb_size - 1
            emb = F.one_hot(target, emb_size)
            atom_x.append(emb)
        atom_x = torch.cat(atom_x, -1).float().to(z)
        
        node_emb = self.node_project(atom_x)
        '''

        G = [dgl.DGLGraph() for _ in range(batch_size)]

        num_nodes = ((atom_y[:, :, 0] != -1) & (atom_y[:, :, 0] != 42))
        num_nodes = num_nodes.int().sum(0)
        
        for b, graph in enumerate(G):
            # add all the nodes,
            # but only edges for node
            # right before the end node
            num_node = num_nodes[b].item()

            graph.add_nodes(seq_length)

            if num_node > 12:
                num_edge = ((12 // 2) * 11) + ((num_node - 12) * 12)
            else:
                num_edge = (num_node * (num_node - 1))//2

            edge_start = self.edge_idx[0][:num_edge]
            edge_end   = self.edge_idx[1][:num_edge]
            graph.add_edges(edge_start, edge_end)
            graph.add_edges(edge_end, edge_start)

            graph = dgl.transform.add_self_loop(graph)
            graph.to(z.device)
            graph.ndata['h'] = node_emb[:, b, :]

            G[b] = graph

        G = dgl.batch(G)

        feats = G.ndata['h']
        for l in self.gcn:
            feats = l(G, feats)
        G.ndata['h'] = feats
        G = dgl.unbatch(G)

        node_feats = [g.ndata['h'].unsqueeze(1) for g in G]
        node_feats = torch.cat(node_feats, 1)

        node_emb    = node_feats + pos_emb
        edge_memory = torch.cat([z.unsqueeze(0), node_emb], 0)

        edge_decoder_emb = []
        edge_memory_mask = []
        edge_groups = []
        for i in range(1, seq_length):
            start = max(0, i - 12)
            end   = i
            edge_decoder_emb.append(node_emb[start:end])
            edge_groups.append(end-start)

            memory_mask = torch.ones(end-start, seq_length + 1) * float('-inf')
            memory_mask[:, [0, i+1]] = 0.0

            edge_memory_mask.append(memory_mask)

        edge_decoder_emb = torch.cat(edge_decoder_emb, 0)
        edge_memory_mask = torch.cat(edge_memory_mask, 0).to(z)

        edge_decoder_mask = torch.ones(len(edge_decoder_emb), len(edge_decoder_emb))
        edge_decoder_mask.fill_(float('-inf'))
        edge_decoder_mask = edge_decoder_mask.to(z)

        edge_num = 0
        for n in edge_groups:
            s = edge_num
            e = edge_num + n
            edge_decoder_mask[s:e, s:e] = 0.0
            edge_num = e

        edge_emb = self.edge_decoder(
            tgt = edge_decoder_emb,
            memory = edge_memory,
            tgt_mask = edge_decoder_mask,
            memory_mask = edge_memory_mask,
        )

        edge_preds  = [c(edge_emb) for c in self.edge_classifiers]
        edge_preds  = [pred.view(-1, pred.shape[-1]) for pred in edge_preds]
        edge_target = bond_y.view(-1, bond_y.shape[-1])
        edge_loss   = self.calc_node_loss(edge_preds, edge_target)

        return node_loss + edge_loss

    def generate(self, z, max_nodes=50):

        batch_size = z.shape[0]
        seq_length = max_nodes

        num_nodes = -1*torch.ones(batch_size)
        num_nodes = num_nodes.to(z)

        zero_node  = torch.zeros(batch_size, self.node_input_dim)
        pos_emb    = self.build_inputs(batch_size, 1, 0).to(z)
        zero_input = pos_emb + self.node_project(zero_node).unsqueeze(0)

        decoder_inputs = [zero_input,]

        pred_node_embs  = []
        pred_node_feats = []
        pred_edge_feats = []

        for i in range(seq_length):

            if (num_nodes != -1).all(): break

            decoder_mask = self.generate_square_mask(i + 1).to(z)

            node_emb = self.node_decoder(
                tgt = torch.cat(decoder_inputs, 0),
                memory = z.unsqueeze(0),
                tgt_mask = decoder_mask,
            )

            node_emb = node_emb[-1]

            # pred_node_embs.append(node_emb.unsqueeze(0))

            node_pred = [c(node_emb) for c in self.node_classifiers]
            node_pred = [F.softmax(pred, -1) for pred in node_pred]

            cond1 = node_pred[0].argmax(1) == self.end_node
            cond2 = num_nodes == -1
            num_nodes[cond1 & cond2] = i #+ 1

            node_emb = []
            for j, emb_size in enumerate(self.one_hot_dims):
                target = node_pred[j].argmax(-1)
                emb    = F.one_hot(target, emb_size)
                node_emb.append(emb)
            node_emb = torch.cat(node_emb, -1).float().to(z)

            node_pred = torch.cat(node_pred, -1)
            pred_node_feats.append(node_pred.unsqueeze(1))

            node_emb = self.node_project(node_emb).unsqueeze(0)
            pred_node_embs.append(node_emb + pos_emb)

            pos_emb  = self.build_inputs(batch_size, 1, i+1).to(z)
            next_input = pos_emb + node_emb # Pos emb here will be 1,2,3...

            decoder_inputs.append(next_input)

            '''
            if i == 0:
                continue

            start = max(0, i - 12)
            end   = i

            edge_emb = self.edge_decoder(
                tgt = torch.cat(pred_node_embs[start:end], 0),
                memory = torch.cat([z.unsqueeze(0), pred_node_embs[-1]], 0)
            )

            edge_preds = [c(edge_emb) for c in self.edge_classifiers]
            edge_preds = [F.softmax(pred, -1) for pred in edge_preds]
            edge_preds = torch.cat(edge_preds, -1)

            pred_edge_feats.append(edge_preds)
            '''

        num_nodes[num_nodes == -1] = len(pred_node_feats)

        seq_length = int(max(num_nodes).item())
        pos_emb    = self.build_inputs(batch_size, seq_length).to(z)

        # pred_node_embs = torch.cat(decoder_inputs[1:], 0)
        pred_node_embs = torch.cat(pred_node_embs, 0)

        G = [dgl.DGLGraph() for _ in range(batch_size)]

        for b, graph in enumerate(G):
            num_node = int(num_nodes[b].item())

            graph.add_nodes(num_node)
            graph.to(z.device)

            if num_node > 12:
                num_edge = ((12 // 2) * 11) + ((num_node - 12) * 12)
            else:
                num_edge = (num_node * (num_node - 1))//2

            edge_start = self.edge_idx[0][:num_edge]
            edge_end   = self.edge_idx[1][:num_edge]
            graph.add_edges(edge_start, edge_end)
            graph.add_edges(edge_end, edge_start)

            graph = dgl.transform.add_self_loop(graph)
            graph.to(z.device)
            graph.ndata['h'] = pred_node_embs[:num_node, b, :]

            G[b] = graph


        G = dgl.batch(G)

        feats = G.ndata['h']
        for l in self.gcn:
            feats = l(G, feats)
        G.ndata['h'] = feats
        G = dgl.unbatch(G)

        node_embs  = torch.zeros(seq_length, len(G), pred_node_embs.shape[-1])
        for b, graph in enumerate(G):
            num_node = int(num_nodes[b].item())
            node_embs[:num_node, b, :] = graph.ndata['h']

        node_embs   = node_embs + pos_emb
        edge_memory = torch.cat([z.unsqueeze(0), node_embs], 0)

        edge_decoder_emb = []
        edge_memory_mask = []
        edge_groups = []
        for i in range(1, seq_length):
            start = max(0, i - 12)
            end   = i
            edge_decoder_emb.append(node_embs[start:end])
            edge_groups.append(end-start)

            memory_mask = torch.ones(end-start, seq_length + 1) * float('-inf')
            memory_mask[:, [0, i+1]] = 0.0

            edge_memory_mask.append(memory_mask)

        edge_decoder_emb = torch.cat(edge_decoder_emb, 0)
        edge_memory_mask = torch.cat(edge_memory_mask, 0).to(z)

        edge_decoder_mask = torch.ones(len(edge_decoder_emb), len(edge_decoder_emb))
        edge_decoder_mask.fill_(float('-inf'))
        edge_decoder_mask = edge_decoder_mask.to(z)

        edge_num = 0
        for n in edge_groups:
            s = edge_num
            e = edge_num + n
            edge_decoder_mask[s:e, s:e] = 0.0
            edge_num = e

        # print(edge_decoder_emb.shape)
        # print(edge_memory.shape)
        # print(edge_decoder_mask.shape)
        # print(edge_memory_mask.shape)

        edge_emb = self.edge_decoder(
            tgt = edge_decoder_emb,
            memory = edge_memory,
            tgt_mask = edge_decoder_mask,
            memory_mask = edge_memory_mask,
        )

        edge_preds = [c(edge_emb) for c in self.edge_classifiers]
        edge_preds = [F.softmax(pred, -1) for pred in edge_preds]
        edge_preds = torch.cat(edge_preds, -1)

        pred_node_feats = torch.cat(pred_node_feats, 1)
        pred_edge_feats = edge_preds

        G = [dgl.DGLGraph() for _ in range(batch_size)]

        for b, graph in enumerate(G):

            num_node = int(num_nodes[b].item())

            graph.add_nodes(num_node)
            graph.ndata['feats'] = pred_node_feats[b, :num_node]

            if num_node > 12:
                num_edge = ((12 // 2) * 11) + ((num_node - 12) * 12)
            else:
                num_edge = (num_node * (num_node - 1))//2

            edge_start = []
            edge_end   = []
            for i in range(num_node):
                for j in range(i):
                    if ((i > 12) and (j < (i - 12))): continue
                    edge_start.append(i)
                    edge_end.append(j)

            graph.add_edges(edge_start, edge_end)
            graph.edata['feats'] = pred_edge_feats[:num_edge, b]

        G = dgl.batch(G)
        G.to(z.device)

        return G
