import dgl
import torch
import numpy as np

from torch import nn
from dgl import function as fn
from data.utils import Library

eps = np.finfo(np.float32).eps.item()

class RewardLoss(nn.Module):

    def __init__(self, device=torch.device('cpu'), discount_factor=0.1):
        super().__init__()
        self.device = device
        self.discount_factor=discount_factor
        self.pool = dgl.nn.pytorch.glob.AvgPooling()

    def node_feat_gen(self, nodes):
        ndata = nodes.data['feats']
        atom_types = ndata[:, :43].argmax(1).data.cpu()
        charges    = ndata[:, 43:50].argmax(1).data.cpu() - 3
        aromatics  = ndata[:, -2:].argmax(1).data.cpu()
        max_bonds = torch.Tensor(Library.max_bonds)[atom_types]
        return {
            'atoms': atom_types,
            'charges': charges,
            'aromatics': aromatics,
            'max_bonds': max_bonds
        }

    def edge_feat_gen(self, edges):
        edata = edges.data['feats']
        bond_types = edata[:, :5].argmax(1).data.cpu()
        return {
            'bt' : bond_types,
        }

    def aromatic_edge_msg(self, edges):
        bt        = edges.data['bt']
        src_atoms = edges.src['atoms']
        dst_atoms = edges.dst['atoms']
        is_aromatic_edge = (src_atoms != 42)  & (bt == 4) & (dst_atoms != 42)
        return {
            'is_aromatic_edge': is_aromatic_edge.float(),
        }

    def aromatic_check(self, nodes):
        check = (nodes.data['has_aromatic_edge'] > 0) ^ (nodes.data['aromatics'] > 0)
        return {'aromatic_fail': check.float()}

    def renumber_aromatic_bond(self, edges):
        bond_types = edges.data['bt']
        bond_types[bond_types == 4] = 1
        # ignore every bond that comes from end node
        bond_types[edges.src['atoms'] == 42] = 0
        # ignore selfloops
        bond_types[edges.data['isgen'] == -1] = 0
        return {'bt': bond_types.float()}

    def calc_valency(self, nodes):
        ndata = nodes.data
        max_bonds = ndata['max_bonds']
        charge = ndata['charges']
        num_bonds = ndata['num_bonds']
        valency = num_bonds - charge
        return {'valency': valency.float()}

    def calc_node_reward(self, nodes):
        ndata = nodes.data

        aromatic_fail = ndata['aromatic_fail'].float()
        valency_fail  = (ndata['valency'] > ndata['max_bonds']).float()
        bondnum_fail = ((ndata['atoms'] == 42) ^ (ndata['num_bonds'] == 0)).float()

        aromatic_pass = (1 - aromatic_fail)
        valency_pass  = (1 - valency_fail)
        bondnum_pass  = (1 - bondnum_fail)

        aromatic_reward = - 1*aromatic_fail
        valency_reward  = - 2*valency_fail
        bondnum_reward  = - 3*bondnum_fail

        reward = aromatic_reward + valency_reward + bondnum_reward

        reward[(ndata['num_nodes'] == 1) & (ndata['atoms'] == 42)] = -4

        reward = reward.to(self.device)

        return {'reward': reward }

    def forward(self, G):

        G.apply_nodes(self.node_feat_gen)
        G.apply_edges(self.edge_feat_gen)

        G.update_all(self.aromatic_edge_msg,
                     fn.sum('is_aromatic_edge', 'has_aromatic_edge'),
                     self.aromatic_check)

        G.apply_edges(self.renumber_aromatic_bond)
        G.update_all(fn.copy_e('bt', '_bt'),
                     fn.sum('_bt', 'num_bonds'),
                     self.calc_valency)

        num_nodes = torch.Tensor(G.batch_num_nodes)
        G.ndata['num_nodes'] = dgl.broadcast_nodes(G, num_nodes)
        G.apply_nodes(self.calc_node_reward)

        # NOTE: the 'reward' system is a little broken.
        # Rewards are based on future edges as well
        G.update_all(fn.copy_e('logprobs', '_lp'),
                     fn.mean('_lp', 'sum_edge_logprobs'),
                     lambda x: {'total_logprobs': x.data['logprobs']+x.data['sum_edge_logprobs']})

        G.apply_nodes(lambda x: {'loss': -x.data['total_logprobs'] * x.data['reward'].detach()})
        total_loss = self.pool(G, G.ndata['loss']).mean()
        total_reward = self.pool(G, G.ndata['reward']).mean()
        return total_loss, total_reward

        total_loss = 0
        total_reward = 0
        G_list = dgl.unbatch(G)
        for g in G_list:
            reward  = g.ndata['reward']
            logprob = g.ndata['total_logprobs']

            node_list = g.nodes()

            if self.discount_factor > 0:
                prev_reward = 0
                for i in reversed(range(len(reward))):
                    reward[i] = reward[i] + (prev_reward * self.discount_factor)
                    prev_reward = reward[i]

            reward = reward.float()
            # reward -= reward.mean()
            # reward /= (reward.std() + eps)
            # reward[reward != reward] = 0 # NaN

            loss = -logprob * reward.detach()

            loss = loss.mean()
            reward = reward.mean()

            total_loss += loss
            total_reward += reward

        total_loss /= len(G_list)
        total_reward /= len(G_list)

        return total_loss, total_reward
