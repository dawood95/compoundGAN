import torch
import random

from .selfies import SELFIES

class ZINC250K(SELFIES):

    def __init__(self, data_file):
        super().__init__(data_file)

        self.condition_dim = 3
        
    def get_data_from_dataframe(self):
        smiles = [x.strip() for x in self.df['smiles']]
        logP   = [float(x) for x in self.df['logP']]
        qed    = [float(x) for x in self.df['qed']]
        sas    = [float(x) for x in self.df['SAS']]

        data = list(zip(smiles, logP, qed, sas))

        random.shuffle(data)

        return data

    
    def __getitem__(self, idx):

        smiles, logP, qed, sas = self.data[idx]
        data_item = self.get_selfies_from_smiles(smiles)

        emb, selfie_tensor, stereo_tensor = data_item
        condition = torch.Tensor([logP, qed * 10, sas])

        return emb[1:-1], emb[:-1], selfie_tensor[1:], stereo_tensor[1:], condition
