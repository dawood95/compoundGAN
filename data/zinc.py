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
        atom_feats.append(af.unsqueeze(0))
        bond_feats.append(bf.unsqueeze(0))

    atom_feats = torch.cat(atom_feats, 0)
    bond_feats = torch.cat(bond_feats, 0)
    graphs     = dgl.batch(graphs)

    return graphs, atom_feats, bond_feats
