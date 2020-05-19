import os
import sys

import torch
import random
import pandas as pd
import numpy as np

from tqdm import tqdm
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import RDConfig
from rdkit.Chem import Descriptors

sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

from torch.utils import data
from torch.nn import functional as F

from selfies import encoder as selfie_encoder
from selfies import selfies_alphabet

SELFIES_STEREO = ['', '\\', '/']

SELFIE_VOCAB = [

    '[START]', '[END]',

    '[Branch1_1]', '[Branch1_2]', '[Branch1_3]',
    '[Branch2_1]', '[Branch2_2]', '[Branch2_3]',
    '[Branch3_1]', '[Branch3_2]', '[Branch3_3]',

    '[Ring1]', '[Ring2]', '[Ring3]',
    '[ExplRing1]', '[ExplRing2]', '[ExplRing3]',
    '[Expl-Ring1]', '[Expl-Ring2]', '[Expl-Ring3]',
    '[Expl=Ring1]', '[Expl=Ring2]', '[Expl=Ring3]',

    '[C]', '[=C]', '[#C]',
    '[C@expl]', '[C@@expl]', '[C@Hexpl]', '[C@@Hexpl]',
    '[C-expl]', '[C+expl]', '[CHexpl]', '[CH-expl]', '[CH+expl]',
    '[=C-expl]',
    '[CH2expl]', '[CH2-expl]', '[CH2+expl]',
    '[#C-expl]',

    '[O]', '[Oexpl]', '[O-expl]',  '[O+expl]', '[OH+expl]',
    '[#O+expl]',
    '[=O]', '[=O+expl]', '[=OH+expl]',

    '[N]', '[Nexpl]', '[N+expl]', '[N-expl]',
    '[NHexpl]', '[NH+expl]', '[NH-expl]',
    '[NH2+expl]', '[NH3+expl]', 
    '[=N]', '[=N+expl]', '[=N-expl]', '[=NH+expl]', '[=NH2+expl]',
    '[#N]', '[#N+expl]',

    '[P]', '[=P]', '[P-expl]', '[P+expl]',
    '[P@@Hexpl]', '[P@@expl]', '[P@expl]',
    '[PHexpl]', '[PH+expl]', '[PH2+expl]', 
    '[=P@@expl]', '[=P@expl]', '[=PHexpl]', '[=PH2expl]',

    '[S]', '[S+expl]', '[S-expl]',
    '[SHexpl]', '[SH+expl]', '[SH-expl]',
    '[S@@+expl]', '[S@@expl]', '[S@expl]',
    '[#S]',
    '[=S]', '[=S+expl]', '[=SHexpl]', '[=SH+expl]',
    '[=S@@expl]', '[=S@expl]', 

    '[H]',
    '[F]', '[F-expl]', '[F+expl]',

    '[I]',
    '[I-expl]', '[I+expl]', '[I+2expl]', '[I+3expl]',
    '[IH2expl]',
    '[=I]', '[=IH2expl]',
    
    '[Br]', '[Br-expl]', '[Br+expl]', '[Br+2expl]', '[Br+3expl]',
    '[Cl]', '[Cl-expl]', '[Cl+expl]', '[Cl+2expl]','[Cl+3expl]',

    '[B]', '[Bexpl]', '[B-expl]',
    '[BH-expl]', '[BH2-expl]', '[BH3-expl]',
    '[=B]', '[=B-expl]',
    
    '[Seexpl]', '[Se-expl]', '[Se+expl]',
    '[SeHexpl]', '[SeH2expl]',
    '[=Seexpl]', '[=Se+expl]',

    '[Siexpl]', '[Si-expl]',
    '[SiHexpl]', '[SiH-expl]', '[SiH2expl]',
    '[=Siexpl]',

    '[epsilon]',
]

class SELFIES(data.Dataset):

    def __init__(self, data_file):
        super().__init__()

        # Read data file
        df = Path(data_file).as_posix()
        df = pd.read_csv(df)
        
        self.df   = df
        self.data = self.get_data_from_dataframe()

        self.condition_dim = 1
        
        return
        
    def get_data_from_dataframe(self):
        
        smiles = self.df['smiles']
        smiles = [x.strip() for x in smiles]

        # Shuffle smiles
        random.shuffle(smiles)
        
        return smiles

    @staticmethod
    def get_selfies_from_smiles(smiles, returnMol=False):
        molecule = Chem.MolFromSmiles(smiles)
        Chem.Kekulize(molecule)
        smiles   = Chem.MolToSmiles(molecule, kekuleSmiles=True)

        selfies  = selfie_encoder(smiles)
        selfies  = '[START]' + selfies + '[END]'
        selfies  = selfies.replace('][', '],[')
        selfies  = selfies.split(',')

        stereo_idx = []
        selfie_idx = []
        for token in selfies:
            if '\\' in token:
                stereo_idx.append(1)
            elif '/' in token:
                stereo_idx.append(2)
            else:
                stereo_idx.append(0)

            token = token.replace('\\', '').replace('/', '')
            selfie_idx.append(SELFIE_VOCAB.index(token))

        selfie_tensor = torch.Tensor(selfie_idx).long()
        stereo_tensor = torch.Tensor(stereo_idx).long()

        selfie_data = F.one_hot(selfie_tensor, len(SELFIE_VOCAB))
        stereo_data = F.one_hot(stereo_tensor, 3)

        emb  = torch.cat([selfie_data, stereo_data], -1)
        
        if returnMol:
            return emb, selfie_tensor, stereo_tensor, molecule
        
        return emb, selfie_tensor, stereo_tensor
    
    def __getitem__(self, idx):

        smiles    = self.data[idx]
        data_item = self.get_selfies_from_smiles(smiles, True)

        emb, selfie_tensor, stereo_tensor, molecule = data_item        

        # Get conditions and normalize in the 0 to 1 range
        logP = Descriptors.MolLogP(molecule)
        tpsa = Descriptors.TPSA(molecule, includeSandP=True) / 10.0

        sa_score = sascorer.calculateScore(molecule)
        
        condition = torch.Tensor([sa_score,])
                
        return emb[1:-1], emb[:-1], \
            selfie_tensor[1:], stereo_tensor[1:], \
            condition

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate(x):
        batch_size = len(x)

        embeddings, token_x, selfie_y, stereo_y, condition = map(list, zip(*x))

        max_seq_len = max(embeddings, key=len).shape[0]

        batch_emb_shape = (max_seq_len, batch_size, embeddings[-1].shape[-1])
        batch_emb       = torch.zeros(batch_emb_shape, dtype=torch.float)
        batch_emb_mask  = torch.zeros(batch_emb_shape[:-1], dtype=torch.bool)
        for b, emb in enumerate(embeddings):
            batch_emb[:emb.shape[0], b] = emb
            batch_emb_mask[:emb.shape[0], b] = True

        max_seq_len = max(token_x, key=len).shape[0]

        batch_token_shape = (max_seq_len, batch_size, token_x[-1].shape[-1])
        batch_token_x     = torch.zeros(batch_token_shape, dtype=torch.float)
        for b, token in enumerate(token_x):
            batch_token_x[:token.shape[0], b] = token

        batch_token_y = -torch.ones((max_seq_len, batch_size, 2)).long()
        for b in range(batch_size):
            batch_token_y[:selfie_y[b].shape[0], b, 0] = selfie_y[b]
            batch_token_y[:selfie_y[b].shape[0], b, 1] = stereo_y[b]

        condition = [x.unsqueeze(0) for x in condition]
        batch_condition = torch.cat(condition, 0).float()

        return batch_emb, batch_emb_mask, batch_token_x, batch_token_y, \
            batch_condition
