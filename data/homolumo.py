import torch
import random

from .selfies import SELFIES

class HOMOLUMO(SELFIES):

    def __init__(self, data_file):
        super().__init__(data_file)

        self.condition_dim = 2
        
    def get_data_from_dataframe(self):
        smiles = [x.strip() for x in self.df['smiles']]
        homo   = [float(x) for x in self.df['homo']]
        lumo   = [float(x) for x in self.df['lumo']]

        data = list(zip(smiles, homo, lumo))

        random.shuffle(data)

        return data

    
    def __getitem__(self, idx):

        smiles, homo, lumo = self.data[idx]
        data_item = self.get_selfies_from_smiles(smiles)

        emb, selfie_tensor, stereo_tensor = data_item
        condition = torch.Tensor([homo, lumo])

        return emb[1:-1], emb[:-1], selfie_tensor[1:], stereo_tensor[1:], condition
