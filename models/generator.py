import torch
from torch import nn
from torch.nn import functional as F

from time import time
from .decoder_cell import DecoderCell

class Generator(nn.Module):
    '''
    z --> Graph
    '''
    def __init__(self, latent_size, node_feats, edge_feats, num_layers=4, bias=True):
        super().__init__()

        node_hidden_size = 256
        node_emb_size = 128
        edge_emb_size = 64
        self.latent_project = nn.Linear(latent_size, node_hidden_size, bias=True)

        self.node_cell = DecoderCell(node_hidden_size, node_emb_size, node_hidden_size,
                                     num_layers, bias)
        self.edge_cell = DecoderCell(node_emb_size, edge_emb_size, node_hidden_size,
                                     num_layers, bias)

        self.node_classifiers = nn.ModuleList()
        for feat_size in node_feats:
            self.node_classifiers.append(nn.Linear(node_emb_size, feat_size, bias))

        self.edge_classifiers = nn.ModuleList()
        for feat_size in edge_feats:
            self.edge_classifiers.append(nn.Linear(edge_emb_size, feat_size, bias))

        self.node_inp_size = node_hidden_size

    def forward(self, z, max_nodes=500):
        raise NotImplementedError

    def calc_loss(self, z, atom_target, bond_target):
        max_seq_length = atom_target.shape[0]
        batch_size = atom_target.shape[1]

        # Node and Edge Loss
        node_loss = 0
        edge_loss = 0
        edge_num = 0
        node_num = 0

        # Project latent space to input dimension
        z = self.latent_project(z)

        # Reset node lstm state
        self.node_cell.reset_hidden(batch_size)
        self.node_cell.set_context(z)

        node_embeddings = []
        x = torch.zeros((batch_size, self.node_inp_size))
        if next(self.latent_project.parameters()).is_cuda:
            x = x.cuda()

        for i in range(max_seq_length):
            # Generate a node
            node_pred = []
            node_emb = self.node_cell(x)
            for c in self.node_classifiers:
                node_pred.append(c(node_emb))

            if i == 0:
                node_embeddings.append(node_emb)
                # might be wrong, maybe concat node output emb ?
                x = self.node_cell.hidden[1][-1]
                continue

            # Generate edges
            edge_preds = [[], [], [], []]
            self.edge_cell.reset_hidden(batch_size)
            self.edge_cell.set_context(self.node_cell.hidden[1][-1])
            for prev_node_emb in node_embeddings[::-1]:
                # can speed this up ?
                _edge_emb = self.edge_cell(prev_node_emb)
                for j, c in enumerate(self.edge_classifiers):
                    edge_preds[j].insert(0, c(_edge_emb))
            edge_preds = [[e.unsqueeze(1) for e in ep] for ep in edge_preds]
            edge_preds = [torch.cat(ep, 1) for ep in edge_preds]

            # Calculate loss
            node_y = atom_target[i]
            for j in range(node_y.shape[1]):
                node_loss += F.cross_entropy(node_pred[j], node_y[:, j], ignore_index=-1)
                node_num += 1

            edge_y = bond_target[i]
            for j in range(edge_y.shape[2]):
                target = edge_y[:, :, j].view(-1)
                pred = edge_preds[j].view(-1, edge_preds[j].shape[-1])
                edge_loss += F.cross_entropy(pred, target,
                                             ignore_index=-1)
                edge_num += 1

            # Store current node embedding
            node_embeddings.append(node_emb)

            # Set next node input as current edge state
            x = self.edge_cell.hidden[1][-1]

        pred_loss = node_loss/node_num + edge_loss/edge_num
        return None, pred_loss


