import dgl
import torch
import pandas as pd

from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from torch.utils import data

from .utils import mol2graph

class ZINC250K(data.Dataset):

    def __init__(self, data_file):
        super().__init__()
        data_file = Path(data_file).as_posix()
        data = pd.read_csv(data_file)
        data = data['smiles']
        self.data = data

    def __getitem__(self, idx):
        smiles = self.data[idx]
        mol = Chem.MolFromSmiles(smiles)
        G, atom_feats, bond_feats = mol2graph(mol)
        return G, atom_feats, bond_feats

    def __len__(self):
        return len(self.data)

def ZINC_collate(x):
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

    bond_target = [[],]*mask.shape[1]
    for i in range(mask.shape[1]):
        t = []
        for b in range(mask.shape[0]):
            if mask[b, i] == 1:
                t.append(bond_feats[b][i])
            else:
                t.append(None)
        _len = 0
        for _t in t:
            if _t is None: continue
            _len = len(_t)
            break
        for j in range(len(t)):
            if t[j] == None:
                t[j] = [[-1,-1,-1,-1],]*_len
        bond_target[i] = t
    bond_target = [torch.Tensor(b).long() for b in bond_target]

    return graphs, atom_targets.long(), bond_target
