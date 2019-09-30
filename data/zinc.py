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

#@profile
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

    atom_targets = torch.ones((max_seq_len, len(atom_feats), af.shape[-1]))
    atom_targets = -1 * atom_targets
    for b, af in enumerate(atom_feats):
        atom_targets[:len(af), b] = af

    bond_targets = torch.ones((max_seq_len, len(bond_feats), max_seq_len-1, bf.shape[-1]))
    bond_targets = -1 * bond_targets
    for b, bf in enumerate(bond_feats):
        bond_targets[:bf.shape[0], b, :bf.shape[1]] = bf

    return graphs, atom_targets.long(), bond_targets.long()
