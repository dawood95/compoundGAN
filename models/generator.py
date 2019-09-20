import torch
from torch import nn
from torch.nn import functional as F

from .decoder_cell import DecoderCell

class Generator(nn.Module):
    '''
    z --> Graph
    '''
    def __init__(self, latent_size, node_feats, edge_feats, num_layers=4, bias=True):
        super().__init__()

        hidden_size = 256
        self.latent_project = nn.Linear(latent_size, hidden_size, bias=True)

        self.node_cell = DecoderCell(hidden_size, hidden_size, num_layers, bias)
        self.edge_cell = DecoderCell(hidden_size, hidden_size, num_layers, bias)

        self.node_classifiers = nn.ModuleList()
        for feat_size in node_feats:
            self.node_classifiers.append(nn.Linear(hidden_size, feat_size, bias))

        self.edge_classifiers = nn.ModuleList()
        for feat_size in edge_feats:
            self.edge_classifiers.append(nn.Linear(hidden_size, feat_size, bias))

        self.end_node = node_feats[0]-1
        self.node_inp_size = hidden_size

    def forward(self, z, max_nodes=500):
        batch_size = z.shape[0]
        assert(batch_size == 1)
        
        z = self.latent_project(z)

        self.node_cell.reset_hidden(batch_size)
        self.node_cell.set_context(z)

        x = torch.zeros((batch_size, self.node_inp_size))
        if z.is_cuda: x = x.cuda()

        node_list = []
        edge_list = []
        node_embs = []
        for i in range(max_nodes):
            node_pred = []
            node_emb  = self.node_cell.forward_unit(x)
            for c in self.node_classifiers:
                pred = c(node_emb)
                pred = F.softmax(pred, 1)
                prob, idx = torch.max(pred, 1)
                node_pred.append((idx, prob))

            if node_pred[0][0] == self.end_node:
                break
            
            if i == 0:
                node_embs.append(node_emb)
                x = node_emb

                node_list.append(node_pred)
                edge_list.append([])
                continue

            edge_preds = []
            self.edge_cell.reset_hidden(batch_size)
            self.edge_cell.set_context(self.node_cell.hidden[-1][-1])
            for prev_node_emb in node_embs[::-1]:
                edge_pred = []
                _edge_emb = self.edge_cell(prev_node_emb)
                for j, c in enumerate(self.edge_classifiers):
                    pred = c(_edge_emb)
                    pred = F.softmax(pred, 1)
                    prob, idx = torch.max(pred, 1)
                    edge_pred.append((idx, prob))
                edge_preds.append(edge_pred)

            no_edges = torch.cat([(pred[0][0] == 0) for pred in edge_preds],0)
            no_edges = no_edges.all()
            if no_edges: break

            edge_preds = edge_preds[::-1]

            node_embs.append(node_emb)
            x = _edge_emb
            node_list.append(node_pred)
            edge_list.append(edge_preds)
            
        return node_list, edge_list

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
        if z.is_cuda: x = x.cuda()

        for i in range(max_seq_length):
            # Generate a node
            node_pred = []
            node_emb = self.node_cell.forward_unit(x)
            for c in self.node_classifiers:
                node_pred.append(c(node_emb))

            node_y = atom_target[i]
            for j in range(node_y.shape[1]):
                _node_loss = F.cross_entropy(node_pred[j], node_y[:, j], ignore_index=-1)
                node_loss += _node_loss
                node_num += 1

            if i == 0:
                node_embeddings.append(node_emb)
                # might be wrong, maybe concat node output emb ?
                x = node_emb##self.node_cell.hidden[1][-1]
                continue

            # Generate edges
            edge_preds = [[], [], [], []]
            self.edge_cell.reset_hidden(batch_size)
            self.edge_cell.hidden = self.node_cell.hidden
            #self.edge_cell.set_context(self.node_cell.hidden[-1]) 

            edge_inp = [prev_node_emb.unsqueeze(0) for prev_node_emb in node_embeddings[::-1]]
            edge_inp = torch.cat(edge_inp, 0)
            edge_emb = self.edge_cell.forward_seq(edge_inp)
            for _edge_emb in edge_emb:
                for j, c in enumerate(self.edge_classifiers):
                    edge_preds[j].append(c(_edge_emb).unsqueeze(1))

            '''
            for prev_node_emb in node_embeddings[::-1]:
                # can speed this up ?
                _edge_emb = self.edge_cell(prev_node_emb)
                for j, c in enumerate(self.edge_classifiers):
                    edge_preds[j].append(c(_edge_emb))
            edge_preds = [[e.unsqueeze(1) for e in ep] for ep in edge_preds]
            '''
            edge_preds = [torch.cat(ep[::-1], 1) for ep in edge_preds]

            # Calculate loss
            edge_y = bond_target[i]
            for j in range(edge_y.shape[2]):
                target = edge_y[:, :len(edge_emb), j].contiguous().view(-1)
                pred = edge_preds[j].view(-1, edge_preds[j].shape[-1])
                edge_loss += F.cross_entropy(pred, target,
                                             ignore_index=-1)
                edge_num += 1

            # Store current node embedding
            node_embeddings.append(node_emb)

            # Set next node input as current edge state
            x = _edge_emb#self.edge_cell.hidden[1][-1]

            '''
            if i % 10 == 0:
                x = x.detach()
                self.node_cell.detach()
                node_embeddings = [emb.detach() for emb in node_embeddings]
            '''

        pred_loss = node_loss + edge_loss
        return None, pred_loss


