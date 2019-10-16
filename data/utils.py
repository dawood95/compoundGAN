import dgl
import torch
import math
import random
import numpy as np

from torch.nn import functional as F
from rdkit import Chem

class Library:
    atom_list = ['C', 'N', 'O', 'S', 'F', 'Si', 'P',\
                 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe',\
                 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl',\
                 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', \
                 'Se', 'Ti', 'Zn', 'Li', 'Ge', 'Cu',\
                 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr',\
                 'Cr', 'Pt', 'Hg', 'Pb']
    charge_list = [-3, -2, -1, 0, 1, 2, 3]
    electron_list = [0, 1, 2]
    chirality_list = ['R', 'S']

def onehot_noise(vector, alpha=2.0):
    '''
    alpha: factor to control how far away one-hot goes
    '''
    noise = torch.zeros_like(vector).uniform_()
    noise = (noise - 0.5) / (alpha*vector.shape[-1])
    clamp = random.random() * 1e-3

    vector = vector + noise
    vector = torch.clamp(vector, clamp)
    logits = torch.log(vector)
    vector = F.softmax(logits, -1)
    return vector

def atoms2vec(atoms):
    atom_idx      = []
    charge_idx    = []
    electron_idx  = []
    chirality_idx = []
    aromatic_idx  = []

    for atom in atoms:
        # Element
        try: idx = Library.atom_list.index(atom.GetSymbol())
        except: raise ValueError
        atom_idx.append(idx)

        # Charge
        idx = atom.GetFormalCharge()+3
        charge_idx.append(idx)

        # Radical Electrons
        idx = atom.GetNumRadicalElectrons()
        electron_idx.append(idx)

        # Aromatic
        idx = atom.GetIsAromatic()
        aromatic_idx.append(idx)

        # Chirality
        try: idx = Library.chirality_list.index(atom.GetProp('_CIPCode'))
        except: idx = len(Library.chirality_list)
        chirality_idx.append(idx)

    atom_idx.append(len(Library.atom_list))
    charge_idx.append(3)
    electron_idx.append(0)
    chirality_idx.append(2)
    aromatic_idx.append(0)

    atom_idx      = torch.Tensor(atom_idx).long()
    charge_idx    = torch.Tensor(charge_idx).long()
    electron_idx  = torch.Tensor(electron_idx).long()
    chirality_idx = torch.Tensor(chirality_idx).long()
    aromatic_idx  = torch.Tensor(aromatic_idx).long()

    atom_emb      = F.one_hot(atom_idx, len(Library.atom_list)+1)
    charge_emb    = F.one_hot(charge_idx, len(Library.charge_list))
    electron_emb  = F.one_hot(electron_idx, len(Library.electron_list))
    chirality_emb = F.one_hot(chirality_idx, len(Library.chirality_list)+1)
    aromatic_emb  = F.one_hot(aromatic_idx, 2)

    feats = [atom_emb, charge_emb, electron_emb, chirality_emb, aromatic_emb]
    feats = [onehot_noise(f.float()) for f in feats] # Add one-hot noise
    feats = torch.cat(feats, dim=1)
    return feats

def bonds2vec(bonds):
    bond_idx       = []
    conjugated_idx = []
    ring_idx       = []
    chirality_idx  = []

    bondtype_list  = [Chem.rdchem.BondType.SINGLE,
                      Chem.rdchem.BondType.DOUBLE,
                      Chem.rdchem.BondType.TRIPLE,
                      Chem.rdchem.BondType.AROMATIC]

    chirality_list = ["STEREONONE", "STEREOANY",
                      "STEREOZ", "STEREOE"]

    for bond in bonds:
        bt = bond.GetBondType()
        bs = str(bond.GetStereo())

        try: bt = bondtype_list.index(bt)+1
        except: bt = 0
        bond_idx.append(bt)
        conjugated_idx.append(bond.GetIsConjugated())
        ring_idx.append(bond.IsInRing())
        chirality_idx.append(chirality_list.index(bs))

    # double since bonds are repeated to make graph undirected
    bond_idx = bond_idx * 2
    conjugated_idx = conjugated_idx * 2
    ring_idx = ring_idx * 2
    chirality_idx = chirality_idx * 2

    bond_idx       = torch.Tensor(bond_idx).long()
    conjugated_idx = torch.Tensor(conjugated_idx).long()
    ring_idx       = torch.Tensor(ring_idx).long()
    chirality_idx  = torch.Tensor(chirality_idx).long()

    bond_emb       = F.one_hot(bond_idx, len(bondtype_list)+1)
    conjugated_emb = F.one_hot(conjugated_idx, 2)
    ring_emb       = F.one_hot(ring_idx, 2)
    chirality_emb  = F.one_hot(chirality_idx, len(chirality_list))

    feats = [bond_emb, conjugated_emb, ring_emb, chirality_emb]
    feats = [onehot_noise(f.float()) for f in feats] # Add one-hot noise
    feats = torch.cat(feats, dim=1)

    no_edge_feat = [0, 0, 0, 0]
    no_edge_feat = [torch.Tensor([i,]).long() for i in no_edge_feat]
    no_edge_feat[0] = F.one_hot(no_edge_feat[0], len(bondtype_list)+1)
    no_edge_feat[1] = F.one_hot(no_edge_feat[1], 2)
    no_edge_feat[2] = F.one_hot(no_edge_feat[2], 2)
    no_edge_feat[3] = F.one_hot(no_edge_feat[3], len(chirality_list))

    return feats.float(), no_edge_feat

def mol2graph(mol, canonical=True, max_len=np.inf):
    # Find Carbon index
    if canonical:
        bfs_root = list(Chem.CanonicalRankAtoms(mol)).index(0)
    else:
        carbon_atoms = []
        for i, atom in enumerate(mol.GetAtoms()):
            if atom.GetSymbol() == 'C':
                carbon_atoms.append(i)
        bfs_root = random.choice(carbon_atoms)

    ''' or Dont
    # Add Hydrogen atoms to molecule
    mol = Chem.AddHs(mol)
    '''

    atoms      = list(mol.GetAtoms())
    bonds      = list(mol.GetBonds())
    edge_start = [b.GetBeginAtomIdx() for b in bonds]
    edge_end   = [b.GetEndAtomIdx() for b in bonds]

    # Get feats
    atom_feats = atoms2vec(atoms)
    bond_feats, no_edge_feats = bonds2vec(bonds)

    # Crete graph to find BFS order
    dummyG = dgl.DGLGraph()
    dummyG.add_nodes(len(atoms))
    dummyG.add_edges(edge_start, edge_end)
    dummyG.add_edges(edge_end, edge_start) # 'undirected' graph

    # Get BFS node sequence
    atom_seq = torch.cat(dgl.bfs_nodes_generator(dummyG, bfs_root))
    atom_seq = [i.item() for i in atom_seq]
    if max_len < np.inf:
        max_len = random.randint(1, max_len)
        atom_seq = atom_seq[:max_len]

    # Create BFS-representation graph
    G = dgl.DGLGraph()
    num_nodes = len(atom_seq) + 1
    num_edges = (num_nodes * (num_nodes - 1)) + num_nodes # + for self loop
    node_feats = torch.zeros((num_nodes, atom_feats.shape[-1]))
    edge_feats = torch.zeros((num_edges, bond_feats.shape[-1]))

    # Go through BFS ordering and fill in features
    G.add_nodes(num_nodes)
    edge_num = 0
    for i in range(len(atom_seq)):
        # i + 1 for self loops
        node_feats[i] = atom_feats[atom_seq[i]].clone()
        for j in range(i):
            s = atom_seq[i]
            e = atom_seq[j]
            if dummyG.has_edge_between(s, e):
                bond_id = dummyG.edge_id(s, e)
                edge_feats[edge_num] = bond_feats[bond_id].clone()
                edge_feats[edge_num + 1] = bond_feats[bond_id].clone()
            else:
                dummy_feats = [onehot_noise(f.float()) for f in no_edge_feats]
                dummy_feats = torch.cat(dummy_feats, dim=1)[0]
                edge_feats[edge_num] = dummy_feats.clone()
                edge_feats[edge_num + 1] = dummy_feats.clone()
            G.add_edge(i, j)
            G.add_edge(j, i)
            edge_num += 2

        # dummy_feats = [onehot_noise(f.float()) for f in no_edge_feats]
        # dummy_feats = torch.cat(dummy_feats, dim=1)[0]
        # edge_feats[edge_num] = dummy_feats.clone()
        G.add_edge(i, i)
        edge_num += 1

    # Add feats for stop node
    node_feats[-1] = atom_feats[-1].clone()
    for j in range(len(atom_seq)):
        dummy_feats = [onehot_noise(f.float()) for f in no_edge_feats]
        dummy_feats = torch.cat(dummy_feats, dim=1)[0]
        edge_feats[edge_num] = dummy_feats.clone()
        edge_feats[edge_num+1] = dummy_feats.clone()
        G.add_edge(j, len(atom_seq))
        G.add_edge(len(atom_seq), j)
        edge_num += 2

    # dummy_feats = [onehot_noise(f.float()) for f in no_edge_feats]
    # dummy_feats = torch.cat(dummy_feats, dim=1)[0]
    # edge_feats[edge_num] = dummy_feats.clone()
    G.add_edge(len(atom_seq), len(atom_seq))

    # set feats to graph
    G.ndata['feats'] = node_feats
    G.edata['feats'] = edge_feats

    # NOTE: Unlike before, we don't need to add self-loops here
    # or in graph convolution. Self-loops are intrinsic now.

    return G
