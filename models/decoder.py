import dgl
import torch
import numpy as np

from torch import nn
from torch.nn import functional as F

from dgl.nn.pytorch.conv import GraphConv
from .decoder_cell import DecoderCell

class Decoder(nn.Module):
    '''
    z --> Graph
    '''
    def __init__(self, latent_size, node_feats, edge_feats,
                 num_layers=1, bias=True):

        super().__init__()

        hidden_size = 256
        node_input_size = sum(node_feats) # hidden_size
        edge_input_size = sum(edge_feats) # hidden_size

        self.latent_project = nn.ModuleList()
        for i in range(num_layers):
            self.latent_project.append(nn.Sequential(
                nn.Linear(latent_size, hidden_size, bias=True),
                # nn.BatchNorm1d(hidden_size),
                # nn.SELU(True),
            ))

        self.node_cell = DecoderCell(node_input_size, hidden_size,
                                     num_layers, bias)
        self.edge_cell = DecoderCell(edge_input_size, hidden_size,
                                     num_layers, bias)

        self.edge_gcn = nn.ModuleList([
            GraphConv(hidden_size, hidden_size, activation=nn.ReLU()),
            GraphConv(hidden_size, hidden_size, activation=nn.ReLU()),
            GraphConv(hidden_size, hidden_size, activation=nn.ReLU()),
        ])

        self.node_classifiers = nn.ModuleList()
        self.edge_classifiers = nn.ModuleList()

        for feat_size in node_feats:
            self.node_classifiers.append(nn.Linear(hidden_size, feat_size, bias))
        for feat_size in edge_feats:
            self.edge_classifiers.append(nn.Linear(2*hidden_size, feat_size, bias))

        self.end_node      = node_feats[0] - 1
        self.node_inp_size = node_input_size
        self.one_hot_sizes = node_feats

    def forward(self, z, max_nodes=50):

        batch_size     = z.shape[0]
        max_seq_length = max_nodes

        num_nodes = -1*torch.ones(batch_size)
        num_nodes = num_nodes.to(z)

        # reset context
        self.node_cell.reset_state(batch_size)

        # layerwise latent variable projection
        latent_context = [l(z).unsqueeze(0) for l in self.latent_project]
        latent_context = torch.cat(latent_context, 0)

        # set context
        self.node_cell.set_context(latent_context)

        pred_node_feats = []
        pred_node_embs  = []

        x = torch.zeros((batch_size, self.node_inp_size)).to(z)
        node_inputs = [x,]

        for i in range(max_seq_length):

            # nodeRNN input
            x = node_inputs[-1]

            # Generate a node
            node_emb  = self.node_cell.forward_unit(x)
            pred_node_embs.append(node_emb.unsqueeze(1))

            node_pred = [c(node_emb) for c in self.node_classifiers]
            node_pred = [F.softmax(pred, -1) for pred in node_pred]

            cond1 = node_pred[0].argmax(1) == self.end_node
            cond2 = num_nodes == -1
            num_nodes[cond1 & cond2] = i + 1

            # Generate node embedding for target/pred (teacher forcing)
            node_emb = []
            for j, emb_size in enumerate(self.one_hot_sizes):
                target = node_pred[j].argmax(-1)
                emb    = F.one_hot(target, emb_size)
                node_emb.append(emb)
            node_emb = torch.cat(node_emb, -1).float()
            node_emb = node_emb.to(z)

            node_pred = torch.cat(node_pred, -1)
            pred_node_feats.append(node_pred.unsqueeze(1))

            node_inputs.append(node_emb)

            if (num_nodes != -1).all(): break

        pred_node_feats = torch.cat(pred_node_feats, 1)
        # pred_edge_feats = torch.cat(pred_edge_feats, 0)
        node_embeddings = torch.cat(pred_node_embs, 1)
        num_nodes[num_nodes == -1] = pred_node_feats.shape[1]

        G = [dgl.DGLGraph() for _ in range(batch_size)]

        for b in range(batch_size):

            num_node = int(num_nodes[b].item())

            G[b].add_nodes(num_node)
            G[b].ndata['emb'] = node_embeddings[b, :num_node]
            G[b].ndata['feats'] = pred_node_feats[b, :num_node]

            edge_start = []
            edge_end   = []
            for i in range(num_node):
                for j in range(i):
                    if ((i > 12) and (j < (i - 12))): continue
                    edge_start.append(i)
                    edge_end.append(j)
            G[b].add_edges(edge_start, edge_end)

        G = dgl.batch(G)
        G.to(z.device)

        '''
        feat = G.ndata['emb']
        for l in self.edge_gcn:
            feat = l(G, feat)
        G.ndata['emb'] = feat
        '''

        G.apply_edges(self.predict_edge)

        edge_preds = []
        for i in range(len(self.edge_classifiers)):
            edge_preds.append(G.edata[str(i)])

        G.edata['feats'] = torch.cat(edge_preds, -1)

        return G

    def calc_node_loss(self, node_pred, node_target):
        total_node_loss = 0
        for i in range(node_target.shape[-1]):
            loss = F.cross_entropy(node_pred[i], node_target[:, i],
                                   ignore_index=-1, reduction='mean')
            total_node_loss += loss
        return total_node_loss

    def calc_edge_loss(self, edge_pred, edge_target):
        total_edge_loss = 0
        for i in range(edge_target.shape[-1]):
            loss = F.cross_entropy(edge_pred[i], edge_target[:, i],
                                   ignore_index=-1, reduction='mean')
            total_edge_loss += loss
        return total_edge_loss

    def predict_edge(self, edges):
        src_feats = edges.src['emb']
        dst_feats = edges.dst['emb']
        feats = torch.cat([src_feats, dst_feats], -1)
        feats = [c(feats) for c in self.edge_classifiers]
        preds = {}
        for i in range(len(feats)):
            preds[str(i)] = feats[i]
        return preds

    def calc_loss(self, z, atom_x, atom_y, bond_y, max_nodes=np.inf):

        batch_size     = z.shape[0]
        seq_length     = atom_x.shape[0]
        max_seq_length = min(seq_length, max_nodes)

        # reset context
        self.node_cell.reset_state(batch_size)

        # layerwise latent variable projection
        latent_context = [l(z).unsqueeze(0) for l in self.latent_project]
        latent_context = torch.cat(latent_context, 0)

        # set context
        self.node_cell.set_context(latent_context)

        atom_emb_seq = self.node_cell.forward_seq(atom_x[:max_seq_length])
        atom_emb_seq = atom_emb_seq.view(-1, atom_emb_seq.shape[-1])
        atom_pred    = [c(atom_emb_seq) for c in self.node_classifiers]

        atom_target  = atom_y[:max_seq_length].view(-1, atom_y.shape[-1])
        node_loss    = self.calc_node_loss(atom_pred, atom_target)

        # reshape atom_embeddings
        atom_emb_seq = atom_emb_seq.view(-1, batch_size, atom_emb_seq.shape[-1])

        # build graph to predict edges
        G = [dgl.DGLGraph() for _ in range(batch_size)]

        edge_target = []
        for b in range(batch_size):
            num_node = atom_y[:, b, 0] != -1
            num_node = num_node.sum() - 1
            num_node = min(num_node, max_seq_length)

            G[b].add_nodes(num_node)
            G[b].ndata['emb'] = atom_emb_seq[:num_node, b]

            edge_start = []
            edge_end   = []
            for i in range(num_node):
                for j in range(i):
                    if ((i > 12) and (j < (i - 12))): continue
                    edge_start.append(i)
                    edge_end.append(j)
                j = 0 if i == 0 else (j + 1)
                edge_target.append(bond_y[i, :j, b])
            G[b].add_edges(edge_start, edge_end)

        edge_target = torch.cat(edge_target, 0)

        G = dgl.batch(G)
        G.to(z.device)

        '''
        feat = G.ndata['emb']
        for l in self.edge_gcn:
            feat = l(G, feat)
        G.ndata['emb'] = feat
        '''

        G.apply_edges(self.predict_edge)

        edge_preds = []
        for i in range(len(self.edge_classifiers)):
            edge_preds.append(G.edata[str(i)])

        edge_loss = self.calc_node_loss(atom_pred, atom_target)

        return node_loss + edge_loss

        # Node and Edge Loss
        node_loss = 0
        edge_loss = 0
        edge_num  = 0
        node_num  = 0

        x = torch.zeros((batch_size, self.node_inp_size)).to(z)
        node_embeddings = [x,]

        for i in range(max_seq_length):

            # nodeRNN input
            x = node_embeddings[-1]

            # Generate a node
            node_emb  = self.node_cell.forward_unit(x)
            node_pred = [c(node_emb) for c in self.node_classifiers]

            # Calculate node loss
            node_loss += self.calc_node_loss(node_pred, atom_target[i])
            node_num  += 1

            # Generate node embedding for target/pred (teacher forcing)
            '''
            node_emb = []
            for j, emb_size in enumerate(self.one_hot_sizes):
                target = atom_target[i, :, j].data.cpu().clone()
                target[target == -1] = emb_size -1
                emb = F.one_hot(target, emb_size)
                node_emb.append(emb)
            node_emb = torch.cat(node_emb, -1).float()
            if z.is_cuda: node_emb = node_emb.cuda()
            '''

            # Dont process edge if first node
            if i == 0:
                node_embeddings.append(node_emb)
                continue

            continue

            # Generate edges
            self.edge_cell.set_hidden(self.node_cell.get_hidden())

            # edgeRNN inputs
            prev_node_embeddings = node_embeddings[:0:-1][:12]
            x = [emb.unsqueeze(0) for emb in prev_node_embeddings]
            x = torch.cat(x, 0)

            # Generate edges
            edge_emb_seq  = self.edge_cell.forward_seq(x)
            edge_emb_seq  = edge_emb_seq[list(range(x.shape[0]))[::-1]]
            edge_emb_seq  = edge_emb_seq.view(-1, edge_emb_seq.shape[-1])
            edge_pred_seq = [c(edge_emb_seq) for c in self.edge_classifiers]

            edge_target = bond_target[i, :len(x)].view(-1, bond_target.shape[-1])
            edge_loss  += self.calc_edge_loss(edge_pred_seq, edge_target)
            edge_num   += x.shape[0]

            # Reset node lstm state and set context
            self.node_cell.set_hidden(self.edge_cell.get_hidden())
            node_embeddings.append(node_emb)

            '''
            if i % 12 == 0:
                self.node_cell.detach()
                self.edge_cell.detach()
                node_embeddings = [emb.detach() for emb in node_embeddings]
            '''

        recon_loss = 0
        if node_num > 0:
            recon_loss += node_loss/node_num
        if edge_num > 0:
            recon_loss += edge_loss/(batch_size*edge_num)
        if node_num + edge_num == 0: raise ValueError

        return recon_loss
