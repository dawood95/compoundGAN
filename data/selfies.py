import torch
import random
import pandas as pd
import numpy as np

from tqdm import tqdm
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
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
    '[CH-expl]', '[CH2-expl]',
    # '[c]', '[-c]', '[=c]',

    '[O]', '[=O]',  '[O-expl]',  '[O+expl]', '[=O+expl]', '[=OH+expl]',
    # '[o]', '[o+expl]',

    '[N]', '[N+expl]', '[N-expl]', '[NH+expl]', '[NH-expl]',
    '[NH2+expl]', '[NH3+expl]', '[NHexpl]',
    '[=N]', '[=N+expl]', '[=N-expl]', '[=NH+expl]', '[=NH2+expl]',
    '[#N]', '[#N+expl]',
    # '[n]', '[n-expl]', '[n+expl]', '[nH+expl]', '[nHexpl]',
    # '[-n]', '[-n+expl]', '[=n+expl]',

    '[P]', '[=P]', '[P+expl]', '[P@@Hexpl]', '[P@@expl]', '[P@expl]',
    '[PH+expl]', '[PHexpl]',
    '[=P@@expl]', '[=P@expl]', '[=PH2expl]',

    '[S]', '[S+expl]', '[S-expl]', '[S@@+expl]', '[S@@expl]', '[S@expl]',
    '[=S]', '[=S+expl]', '[=S@@expl]', '[=S@expl]', '[=SH+expl]',
    # '[s]', '[s+expl]',

    '[H]', '[F]', '[I]', '[Br]', '[Cl]', '[epsilon]'

]

class SELFIES(data.Dataset):

    def __init__(self, data_file):
        super().__init__()

        # Read data file
        data = Path(data_file).as_posix()
        data = pd.read_csv(data)
        data = data['smiles']
        data = [x.strip() for x in data]

        # Shuffle data
        random.shuffle(data)

        #
        self.data = data
        return

    def __getitem__(self, idx):

        smiles   = self.data[idx]
        molecule = Chem.MolFromSmiles(smiles)
        logP     = Descriptors.MolLogP(molecule)
        
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

        return emb[1:-1], logP, emb[:-1], selfie_tensor[1:], stereo_tensor[1:]

    def __len__(self):
        return len(self.data)


def collate(x):
    batch_size = len(x)

    embeddings, logP, token_x, selfie_y, stereo_y = map(list, zip(*x))

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

    batch_logP = torch.Tensor(logP).float().unsqueeze(-1)

    return batch_emb, batch_emb_mask, batch_logP, \
           batch_token_x, batch_token_y
