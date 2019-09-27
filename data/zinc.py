import dgl
import os
import torch
import pandas as pd
import numpy as np

from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from torch.utils import data

from .utils import mol2graph, Library

class ZINC250K(data.Dataset):

    def __init__(self, data_file):
        super().__init__()
        data_file = Path(data_file).as_posix()
        data = pd.read_csv(data_file)
        data = list(data['smiles'])
        self.data = data

    def __getitem__(self, idx):
        smiles = self.data[idx]
        mol = Chem.MolFromSmiles(smiles)
        G, atom_feats, bond_feats = mol2graph(mol)
        return G, atom_feats, bond_feats

    def __len__(self):
        return len(self.data)

def ZINC_collate(x):
    #return None, None, None
    graphs = []
    atom_feats = []
    bond_feats = []
    for g, af, bf in x:
        graphs.append(g)
        atom_feats.append(af)
        bond_feats.append(bf)
    graphs = dgl.batch(graphs)

    max_seq_len = 0
    for bn in bond_feats:
        max_seq_len = max([len(bn), max_seq_len])

    mask = []
    # for each item in batch
    for bn in bond_feats:
        _mask = torch.zeros((1, max_seq_len))
        _mask[0, :len(bn)] = 1
        mask.append(_mask)
    mask = torch.cat(mask)

    atom_targets = -1*torch.ones((mask.shape[1], mask.shape[0], 5))
    for i in range(mask.shape[1]):
        for b in range(len(atom_feats)):
            if mask[b, i] == 1:
                atom_targets[i, b, :] = torch.Tensor(atom_feats[b][i])
            else:
                atom_targets[i, b, :] = -1#torch.Tensor([len(Library.atom_list), 3, 0, 2, 0])

    bond_target = torch.zeros((mask.shape[1], mask.shape[0], max_seq_len, 4))
    bond_target[:] = -1
    # not the most efficient, but whatever
    for i in range(mask.shape[1]):
        for b in range(mask.shape[0]):
            if i >= len(bond_feats[b]): continue
            feat = bond_feats[b][i]
            if len(feat) == 0: continue
            bond_target[i, b, :len(feat)] = torch.Tensor(bond_feats[b][i])

    return graphs, atom_targets.long(), bond_target.long()
