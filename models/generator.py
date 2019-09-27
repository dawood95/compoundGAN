import torch
import numpy as np

from torch import nn
from torch.nn import functional as F

from .decoder_cell import DecoderCell

class Generator(nn.Module):
    '''
    z --> Graph
    '''
    def __init__(self, latent_size, node_feats, edge_feats,
                 num_layers=1, bias=True):

        super().__init__()

        hidden_size = 256
        input_size  = sum(node_feats)

        self.latent_project = nn.Linear(latent_size, hidden_size, bias=True)

        self.node_celli     = DecoderCell(input_size, hidden_size, num_layers, bias)
        self.edge_celli     = DecoderCell(input_size, hidden_size, num_layers, bias)

        self.node_classifiers = nn.ModuleList()
        self.edge_classifiers = nn.ModuleList()

        for feat_size in node_feats:
            self.node_classifiers.append(nn.Linear(hidden_size, feat_size, bias))
        for feat_size in edge_feats:
            self.edge_classifiers.append(nn.Linear(hidden_size, feat_size, bias))

        self.end_node      = node_feats[0]-1
        self.node_inp_size = input_size
        self.one_hot_sizes = node_feats

        def init_weights(m):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_normal_(param.data)
                else:
                    nn.init.constant_(param.data, 0)

        #self.latent_project.apply(init_weights)
        #self.node_celli.apply(init_weights)
        #self.node_classifiers.apply(init_weights)
        #self.edge_celli.apply(init_weights)
        #self.edge_classifiers.apply(init_weights)

    def forward(self, z, max_nodes=500):

        batch_size = z.shape[0]
        assert(batch_size == 1)

        z = self.latent_project(z)

        self.node_celli.reset_hidden(batch_size)
        self.node_celli.set_context(z)

        node_list = []
        edge_list = []

        node_embeddings = []
        for i in range(max_nodes):

            if i == 0:
                x = torch.zeros((batch_size, self.node_inp_size))
                if z.is_cuda: x = x.cuda()
            else:
                x = node_embeddings[-1]    

            node_emb  = self.node_celli.forward_unit(x)

            node_pred = []
            for c in self.node_classifiers:
                pred = c(node_emb)
                pred = F.softmax(pred, 1)
                prob, idx = torch.max(pred, 1)
                node_pred.append((idx, prob))

            if node_pred[0][0] == self.end_node:
                break

            node_emb = []
            for j, emb_size in enumerate(self.one_hot_sizes):
                emb = F.one_hot(node_pred[j][0], emb_size)
                node_emb.append(emb)
            node_emb = torch.cat(node_emb, -1).float()
            if z.is_cuda: node_emb = node_emb.cuda()
            node_embeddings.append(node_emb)
            
            if i == 0:
                node_list.append(node_pred)
                edge_list.append([])
                continue

            edge_preds = []
            self.edge_celli.reset_hidden(batch_size)
            self.edge_celli.hidden = self.node_celli.hidden

            edge_inp = [prev_nemb.unsqueeze(0) for prev_nemb in node_embeddings[::-1]]
            edge_inp = torch.cat(edge_inp, 0)
            edge_emb = self.edge_celli.forward_seq(edge_inp)
            for _edge_emb in edge_emb:
                edge_pred = []
                for j, c in enumerate(self.edge_classifiers):
                    pred = c(_edge_emb)
                    pred = F.softmax(pred, 1)
                    prob, idx = torch.max(pred, 1)
                    edge_pred.append((idx, prob))
                edge_preds.append(edge_pred)
            edge_preds = edge_preds[::-1]
            
            no_edges = torch.cat([(pred[0][0] == 0) for pred in edge_preds],0)
            no_edges = no_edges.all()
            if no_edges: break

            self.node_celli.reset_hidden(batch_size)
            self.node_celli.hidden = self.edge_celli.hidden
            
            node_list.append(node_pred)
            edge_list.append(edge_preds)

        return node_list, edge_list

    def calc_loss(self, z, atom_target, bond_target, max_steps=np.inf):

        max_seq_length = min(atom_target.shape[0], max_steps)
        batch_size     = atom_target.shape[1]

        # Node and Edge Loss
        node_loss = 0
        edge_loss = 0
        edge_num  = 0
        node_num  = 0

        # Project latent space to input dimension
        # NOTE: Maybe don't need it
        z = self.latent_project(z)

        # Reset node lstm state and set context
        self.node_celli.reset_hidden(batch_size)
        self.node_celli.set_context(z)

        node_embeddings = []
        for i in range(max_seq_length):

            # Create input for node RNN
            if i == 0:
                x = torch.zeros((batch_size, self.node_inp_size))
                if z.is_cuda: x = x.cuda()
            else:
                x = node_embeddings[-1]

            # Generate a node
            node_emb  = self.node_celli.forward_unit(x)
            node_pred = [c(node_emb) for c in self.node_classifiers]

            # Calculate node loss
            for j in range(len(self.node_classifiers)):
                node_loss += F.cross_entropy(node_pred[j], atom_target[i, :, j])
                node_num  += 1

            # Generate node embedding for target/pred (teacher forcing)
            node_emb = []
            for j, emb_size in enumerate(self.one_hot_sizes):
                emb = F.one_hot(atom_target[i, :, j], emb_size)
                node_emb.append(emb)
            node_emb = torch.cat(node_emb, -1).float()
            if z.is_cuda: node_emb = node_emb.cuda()

            node_embeddings.append(node_emb)

            # Dont process edge if first node
            if i == 0: continue

            # Generate edges
            edge_preds = [[], [], [], []]
            self.edge_celli.reset_hidden(batch_size)
            self.edge_celli.hidden = self.node_celli.hidden

            edge_inp = [prev_nemb.unsqueeze(0) for prev_nemb in node_embeddings[::-1]]
            edge_inp = torch.cat(edge_inp, 0)
            edge_emb = self.edge_celli.forward_seq(edge_inp)
            for _edge_emb in edge_emb:
                for j, c in enumerate(self.edge_classifiers):
                    edge_preds[j].append(c(_edge_emb).unsqueeze(1))
            edge_preds = [torch.cat(ep[::-1], 1) for ep in edge_preds]

            # Calculate loss
            edge_y = bond_target[i]
            for j in range(edge_y.shape[2]):
                target = edge_y[:, :len(edge_emb), j].contiguous().view(-1)
                pred = edge_preds[j].view(-1, edge_preds[j].shape[-1])
                edge_loss += F.cross_entropy(pred, target,
                                             ignore_index=-1)
                edge_num += 1

            # Reset node lstm state and set context
            self.node_celli.reset_hidden(batch_size)
            self.node_celli.hidden = self.edge_celli.hidden

            '''
            if i % 10 == 0:
                x = x.detach()
                self.node_celli.detach()
                node_embeddings = [emb.detach() for emb in node_embeddings]
            '''

        pred_loss = 0
        if node_num > 0:
            pred_loss += node_loss/node_num
        if edge_num > 0:
            pred_loss += edge_loss/edge_num
        if node_num + edge_num == 0: raise ValueError

        return None, pred_loss
