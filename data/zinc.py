import dgl
import os
import torch
import random
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
        random.shuffle(data)

        self.data = data
        self.seq_length = np.inf

    def __getitem__(self, idx):
        smiles = self.data[idx]
        mol = Chem.MolFromSmiles(smiles)
        item = mol2graph(mol, self.seq_length)
        # G, atom_idx, atom_feats, atom_targets, bond_targets
        return item

    def __len__(self):
        return len(self.data)

#@profile
def ZINC_collate(x):
    batch_size = len(x)

    graphs, atom_idx, atom_x, atom_y, bond_y = map(list, zip(*x))

    graphs = dgl.batch(graphs)

    max_seq_len = 0
    for ay in atom_y:
        max_seq_len = max([len(ay), max_seq_len])

    batch_atom_idx = torch.zeros((max_seq_len, batch_size, atom_idx[-1].shape[-1]))
    batch_atom_x   = torch.zeros((max_seq_len, batch_size, atom_x[-1].shape[-1]))
    batch_atom_y = -1 * torch.ones((max_seq_len, batch_size, atom_y[-1].shape[-1]))

    for b, aidx in enumerate(atom_idx):
        batch_atom_idx[:len(aidx), b] = aidx

    for b, ax in enumerate(atom_x):
        batch_atom_x[:len(ax), b] = ax

    for b, ay in enumerate(atom_y):
        batch_atom_y[:len(ay), b] = ay

    if max_seq_len > 12:
        num_edges = ((12 // 2) * 11) + ((max_seq_len - 12) * 12)
    else:
        num_edges = (max_seq_len * (max_seq_len - 1)) // 2

    batch_bond_y = -1 * torch.ones((num_edges, batch_size, bond_y[-1].shape[-1]))
    for b, by in enumerate(bond_y):
        batch_bond_y[:len(by), b] = by

    return graphs, \
        batch_atom_idx.float(), batch_atom_x.float(), \
        batch_atom_y.long(), batch_bond_y.long()
