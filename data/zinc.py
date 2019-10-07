import dgl
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
        G = mol2graph(mol)
        return G

    def __len__(self):
        return len(self.data)

def ZINC_collate(graphs):
    return dgl.batch(graphs)
