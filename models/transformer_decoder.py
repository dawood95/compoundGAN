import dgl
import torch
import numpy as np

from torch import nn
from torch.nn import functional as F

class Decoder(nn.Module):

    def __init__(self, latent_dim, node_feats, edge_feats, num_layers=1, bias=True,
                 num_head=1):

        super().__init__()

        hidden_dim     = latent_dim
        node_input_dim = sum(node_feats)

        self.node_project = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim, bias=bias),
            nn.BatchNorm1d(hidden_dim),
            nn.SELU(True)
        )

        self.node_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(hidden_dim, num_head, num_head * hidden_dim // 2),
            num_layers,
            nn.LayerNorm(hidden_dim)
        )

        self.edge_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(hidden_dim, num_head, num_head * hidden_dim // 2),
            num_layers // 2,
            nn.LayerNorm(hidden_dim)
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

        self.sin_freq = torch.arange(0, hidden_dim, 2.0) / (hidden_dim)
        self.sin_freq = 1 / (10_000 ** self.sin_freq)

        self.cos_freq = torch.arange(1, hidden_dim, 2.0) / (hidden_dim)
        self.cos_freq = 1 / (10_000 ** self.cos_freq)

    def build_inputs(self, batch_size, seq_length, pos=None):

        embedding = torch.zeros(seq_length, batch_size, self.hidden_dim)

        x = torch.arange(0, seq_length).unsqueeze(-1)
        x = x.repeat_interleave(batch_size, dim=1)
        if pos is not None: x[:] = pos

        sin_emb = torch.sin(torch.einsum('ij,d->ijd', x, self.sin_freq))
        cos_emb = torch.cos(torch.einsum('ij,d->ijd', x, self.cos_freq))

        # sin_emb = sin_emb.unsqueeze(1).repeat_interleave(batch_size, dim=1)
        # cos_emb = cos_emb.unsqueeze(1).repeat_interleave(batch_size, dim=1)

        embedding[:, :, 0:self.hidden_dim:2] = sin_emb.clone().detach()
        embedding[:, :, 1:self.hidden_dim:2] = cos_emb.clone().detach()

        return embedding

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

        atom_x = atom_x.view(-1, atom_x.shape[-1])
        atom_x = self.node_project(atom_x)
        atom_x = atom_x.view(seq_length, batch_size, -1)

        decoder_inputs = self.build_inputs(batch_size, seq_length, atom_i.squeeze(-1))
        decoder_inputs = decoder_inputs.to(z)
        decoder_inputs = decoder_inputs + atom_x

        decoder_mask = self.generate_square_mask(seq_length).to(z)
        # padding_mask = (atom_y[:seq_length, :, 0] == -1).T

        node_emb = self.node_decoder(
            tgt = decoder_inputs,
            memory = z.unsqueeze(0),
            tgt_mask = decoder_mask,
            # tgt_key_padding_mask = padding_mask
        )

        node_preds  = [c(node_emb) for c in self.node_classifiers]
        node_preds  = [pred.view(-1, pred.shape[-1]) for pred in node_preds]
        node_target = atom_y.view(-1, atom_y.shape[-1])

        node_loss = self.calc_node_loss(node_preds, node_target)

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

        edge_preds = [c(edge_emb) for c in self.edge_classifiers]
        edge_preds = [pred.view(-1, pred.shape[-1]) for pred in edge_preds]
        edge_target = bond_y.view(-1, bond_y.shape[-1])

        edge_loss = self.calc_node_loss(edge_preds, edge_target)

        return node_loss + edge_loss

    def generate(self, z, max_nodes=50):

        batch_size = z.shape[0]
        seq_length = max_nodes

        num_nodes = -1*torch.ones(batch_size)
        num_nodes = num_nodes.to(z)

        zero_node  = torch.zeros(batch_size, self.node_input_dim)
        zero_input = self.build_inputs(batch_size, 1, 0).to(z)
        zero_input = zero_input + self.node_project(zero_node).unsqueeze(0)

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
            pred_node_embs.append(node_emb.unsqueeze(0))

            node_pred = [c(node_emb) for c in self.node_classifiers]
            node_pred = [F.softmax(pred, -1) for pred in node_pred]

            cond1 = node_pred[0].argmax(1) == self.end_node
            cond2 = num_nodes == -1
            num_nodes[cond1 & cond2] = i + 1

            node_emb = []
            for j, emb_size in enumerate(self.one_hot_dims):
                target = node_pred[j].argmax(-1)
                emb    = F.one_hot(target, emb_size)
                node_emb.append(emb)
            node_emb = torch.cat(node_emb, -1).float().to(z)

            node_pred = torch.cat(node_pred, -1)
            pred_node_feats.append(node_pred.unsqueeze(1))

            next_input = self.build_inputs(batch_size, 1, i+1).to(z)
            next_input = next_input + self.node_project(node_emb).unsqueeze(0)

            decoder_inputs.append(next_input)

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

        pred_node_feats = torch.cat(pred_node_feats, 1)
        pred_edge_feats = torch.cat(pred_edge_feats, 0)
        num_nodes[num_nodes == -1] = pred_node_feats.shape[1]

        G = [dgl.DGLGraph() for _ in range(batch_size)]

        for b in range(batch_size):

            num_node = int(num_nodes[b].item())

            G[b].add_nodes(num_node)
            G[b].ndata['feats'] = pred_node_feats[b, :num_node]

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

            G[b].add_edges(edge_start, edge_end)
            G[b].edata['feats'] = pred_edge_feats[:num_edge, b]

        G = dgl.batch(G)
        G.to(z.device)

        return G
