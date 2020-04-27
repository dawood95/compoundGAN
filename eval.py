import os
import sys
import argparse
import torch

from tqdm import tqdm
from collections import defaultdict
from multiprocessing import Pool
from matplotlib import pyplot as plt
plt.style.use('ggplot')

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, RDConfig

sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

import selfies
from selfies import encoder as selfies_encoder
from selfies import decoder as selfies_decoder

from data.selfies import SELFIES, SELFIE_VOCAB, SELFIES_STEREO
from models.network import CVAEF

import joypy
import numpy as np
from sklearn.metrics import r2_score

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--data-file', type=str, required=True)
parser.add_argument('--datastat-file', type=str, required=True)
parser.add_argument('--num-mols', type=int, default=1000)
parser.add_argument('--batch-size', type=int, default=32)

parser.add_argument('--token-dim', type=int, default=139)
parser.add_argument('--latent-dim', type=int, default=256)
parser.add_argument('--cnf-dims', type=list, default=[256,256,256,256])

parser.add_argument('--logp', action='store_true')
parser.add_argument('--tpsa', action='store_true')
parser.add_argument('--sascore', action='store_true')

parser.add_argument('--logp-dim', type=int, default=0)
parser.add_argument('--tpsa-dim', type=int, default=1)
parser.add_argument('--sascore-dim', type=int, default=2)

parser.add_argument('--eval-name', type=str, default='eval')

parser.add_argument('--nproc', type=int, default=4)
parser.add_argument('--cuda', action='store_true')

args = parser.parse_args()

condition_dim = 0
if args.logp: condition_dim += 1
if args.tpsa: condition_dim += 1
if args.sascore: condition_dim += 1

DEVICE = torch.device('cuda:0') if args.cuda else torch.device('cpu')

def smiles2mol(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        Chem.SanitizeMol(mol, catchErrors=False)
        smiles = Chem.CanonSmiles(Chem.MolToSmiles(mol))
    except Exception as e:
        print('MOL ERROR : ', e)
        return '', None, False

    try:
        AllChem.AssignAtomChiralTagsFromStructure(
            mol, confId=-1, replaceExistingTags=True
        )

        is_3d_valid = AllChem.EmbedMolecule(
            mol,
            useExpTorsionAnglePrefs=True,
            useBasicKnowledge=True,
            useRandomCoords=True
        )

        AllChem.MMFFOptimizeMolecule(mol)
        is_3d_valid = True
    except Exception as e:
        print ("MOL ERROR: 3D Invalid ", e)
        is_3d_valid = False

    return smiles, mol, is_3d_valid


def calc_properties(mol):
    if (mol is None):
        return None, None, None
    logp    = Descriptors.MolLogP(mol)
    tpsa    = Descriptors.TPSA(mol)
    sascore = sascorer.calculateScore(mol)
    return logp, tpsa, sascore

def calc_data_stats(smiles):
    mol = Chem.MolFromSmiles(smiles)
    Chem.Kekulize(mol)
    smiles = Chem.CanonSmiles(Chem.MolToSmiles(mol))
    _, mol, is_3d_valid = smiles2mol(smiles)
    logp, tpsa, sascore = calc_properties(mol)
    return smiles, is_3d_valid, logp, tpsa, sascore


def pred2smiles(pred, token_dim=args.token_dim):
    tokens = []
    it = zip(
        pred[:, :token_dim].argmax(-1).long().tolist(), # SELFIE token
        pred[:, token_dim:].argmax(-1).long().tolist(), # STEREO token
    )
    for selfie_token_idx, stereo_token_idx in it:
        selfie_token = SELFIE_VOCAB[selfie_token_idx]
        stereo_token = SELFIE_VOCAB[stereo_token_idx]
        if selfie_token_idx > 1:
            selfie_token = '[' + stereo_token + selfie_token[1:]
        tokens.append(selfie_token)
        if selfie_token_idx == 1:
            # END TOKEN
            break

    selfie = ''.join(tokens)
    smiles = selfies.decoder(selfie.replace('[START]', '').replace('[END]', ''))
    smiles, mol, is_3d_valid = smiles2mol(smiles)
    logp, tpsa, sascore      = calc_properties(mol)

    return smiles, mol, is_3d_valid, logp, tpsa, sascore


dataset = SELFIES(args.data_file)

try:
    data_stats = torch.load(args.datastat_file)
except:
    data_stats = []
    with Pool(args.nproc) as pool:
        it = pool.imap_unordered(calc_data_stats, dataset.data)
        it = tqdm(it, total=len(dataset.data))
        for d in it:
            data_stats.append({
                'smiles'      : d[0],
                'is_3d_valid' : d[1],
                'logp'        : d[2],
                'tpsa'        : d[3],
                'sascore'     : d[4]
            })
    torch.save(data_stats, args.datastat_file)

dataset_smiles = [d['smiles'] for d in data_stats]

state_dict = torch.load(args.model)
# Max number of atoms to generate
# Add to argparse if you want to make it dynamic
seq_len    = 300


model = CVAEF(
    [args.token_dim, 3],
    args.latent_dim,
    args.cnf_dims,
    condition_dim,
    1.0, True, use_adjoint=True
)
model.load_state_dict(state_dict['parameters'])
model = model.eval().to(DEVICE)


preds = []
conditions = []
for i in tqdm(range(args.num_mols // args.batch_size)):

    z = torch.zeros((args.batch_size, args.latent_dim - condition_dim))
    z.normal_(0, 1)

    if condition_dim > 0:
        condition = torch.zeros((args.batch_size, condition_dim))
        condition.uniform_(0, 1)

        if args.logp:
            condition[:, args.logp_dim] = ((condition[:, args.logp_dim] - 0.5) * 12).int().float()
        if args.tpsa:
            condition[:, args.tpsa_dim] = (condition[:, args.tpsa_dim] * 16).int().float()
        if args.sascore:
            condition[:, args.sascore_dim] = (condition[:, args.sascore_dim] * 11).int().float()

        z = torch.cat([z, condition], -1)

    z = z.to(DEVICE)

    w = model.cnf(z, None, True)[0]

    pred = model.decoder.generate(z, seq_len)
    for b in range(args.batch_size):
        preds.append(pred[:, b].data.cpu())
        if (condition_dim > 0):
            conditions.append(condition[b, :])

data = []

logp_map = defaultdict(list)
tpsa_map = defaultdict(list)
sascore_map = defaultdict(list)

for i in range(-5, 6): logp_map[i] = []
for i in range(0, 151, 10): tpsa_map[i] = []
for i in range(0, 11): sascore_map[i] = []

for i, d in tqdm(enumerate(map(pred2smiles, preds)), total=len(preds)):
    if d[1] is None: continue
    data.append({
        'smiles'      : d[0],
        'is_3d_valid' : d[2],
        'logp'        : (None if not args.logp else conditions[i][args.logp_dim].item(), d[3]),
        'tpsa'        : (None if not args.tpsa else conditions[i][args.tpsa_dim].item() * 10, d[4]),
        'sascore'     : (None if not args.sascore else conditions[i][args.sascore_dim].item(), d[5]),
    })

    if args.logp:
        logp_map[conditions[i][args.logp_dim].item()].append(d[3])
    if args.tpsa:
        tpsa_map[conditions[i][args.tpsa_dim].int().item() * 10].append(d[4])
    if args.sascore:
        sascore_map[conditions[i][args.sascore_dim].item()].append(d[5])

for d in data_stats:
    logp_map['dataset'].append(d['logp'])
    tpsa_map['dataset'].append(d['tpsa'])
    sascore_map['dataset'].append(d['sascore'])

if args.logp:
    plt.figure()
    joypy.joyplot(logp_map, fade=True, x_range=(-8, 8))
    plt.savefig(args.eval_name+'_logp.png')

    x = []
    for k, v in logp_map.items():
        if k == 'dataset': continue
        for _v in v:
            x.append((float(k), float(_v)))
    x = np.array(x)
    print("LOGP R^2 Score : %.2f"%r2_score(x[:, 0], x[:, 1]))

if args.tpsa:
    plt.figure()
    joypy.joyplot(tpsa_map, fade=True, x_range=(-25, 175))
    plt.savefig(args.eval_name+'_tpsa.png')

    x = []
    for k, v in tpsa_map.items():
        if k == 'dataset': continue
        for _v in v:
            x.append((float(k), float(_v)))
    x = np.array(x)
    print("TPSA R^2 Score : %.2f"%r2_score(x[:, 0], x[:, 1]))

if args.sascore:
    plt.figure()
    joypy.joyplot(sascore_map, fade=True, x_range=(-2, 11))
    plt.savefig(args.eval_name+'_sascore.png')

    x = []
    for k, v in sascore_map.items():
        if k == 'dataset': continue
        for _v in v:
            x.append((float(k), float(_v)))
    x = np.array(x)
    print("SASCORE R^2 Score : %.2f"%r2_score(x[:, 0], x[:, 1]))


pred_smiles = []
novel_smiles = []
num_3d_valid = 0
for d in tqdm(data):
    pred_smiles.append(d['smiles'])
    if d['smiles'] not in set(dataset_smiles):
        novel_smiles.append(d['smiles'])
    if d['is_3d_valid']: num_3d_valid += 1

print("VALIDITY : %d"%(len(pred_smiles)))
print("DIVERSITY : %d"%(len(set(pred_smiles))))
print("NOVELTY : %d"%(len(set(novel_smiles))))
print("3D VALIDITY: %d"%(num_3d_valid))
