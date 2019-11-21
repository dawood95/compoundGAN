import dgl
import torch
import numpy as np

from random import randint
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

def atoms2vec(atoms):
    atom_idx      = []
    charge_idx    = []
    electron_idx  = []
    chirality_idx = []

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

        # Chirality
        try: idx = Library.chirality_list.index(atom.GetProp('_CIPCode'))
        except: idx = len(Library.chirality_list)
        chirality_idx.append(idx)

    atom_idx      = torch.Tensor(atom_idx).long()
    charge_idx    = torch.Tensor(charge_idx).long()
    electron_idx  = torch.Tensor(electron_idx).long()
    chirality_idx = torch.Tensor(chirality_idx).long()

    atom_emb      = F.one_hot(atom_idx, len(Library.atom_list)+1)
    charge_emb    = F.one_hot(charge_idx, len(Library.charge_list))
    electron_emb  = F.one_hot(electron_idx, len(Library.electron_list))
    chirality_emb = F.one_hot(chirality_idx, len(Library.chirality_list)+1)

    feats = [atom_emb, charge_emb, electron_emb, chirality_emb]
    feats = torch.cat(feats, dim=1)

    end_node = torch.Tensor([len(Library.atom_list), 3, 0, 2]).long()
    end_node = end_node.unsqueeze(0)

    target = [atom_idx, charge_idx, electron_idx, chirality_idx]
    target = [t.unsqueeze(1) for t in target]
    target = torch.cat(target, dim=1)
    target = torch.cat([target, end_node], dim=0)

    return feats.float(), target.long()

def bonds2vec(bonds, repeat_bonds=True):
    bond_idx       = []
    conjugated_idx = []
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

        assert bt != 4, 'Kekulization failed maybe ?'

        bond_idx.append(bt)
        conjugated_idx.append(bond.GetIsConjugated())
        chirality_idx.append(chirality_list.index(bs))

    if repeat_bonds:
        bond_idx       = bond_idx * 2
        conjugated_idx = conjugated_idx * 2
        chirality_idx  = chirality_idx * 2

    bond_idx       = torch.Tensor(bond_idx).long()
    conjugated_idx = torch.Tensor(conjugated_idx).long()
    chirality_idx  = torch.Tensor(chirality_idx).long()

    bond_emb       = F.one_hot(bond_idx, len(bondtype_list)+1)
    conjugated_emb = F.one_hot(conjugated_idx, 2)
    chirality_emb  = F.one_hot(chirality_idx, len(chirality_list))

    feats = [bond_emb, conjugated_emb, chirality_emb]
    feats = torch.cat(feats, dim=1)
    feats[feats == 0] = -1

    target = [bond_idx, conjugated_idx, chirality_idx]
    target = [t.unsqueeze(1) for t in target]
    target = torch.cat(target, dim=1)

    return feats.float(), target.long()

#@profile
def mol2graph(mol, num_nodes=np.inf):
    # Find canonical start atom before kek
    bfs_root  = list(Chem.CanonicalRankAtoms(mol))
    num_nodes = min(num_nodes, len(bfs_root) + 1)
    bfs_root  = bfs_root.index(0)

    # Kekulize to remove aromatic flags
    Chem.Kekulize(mol, clearAromaticFlags=True)

    # Get atoms and bonds
    atoms      = list(mol.GetAtoms())
    bonds      = list(mol.GetBonds())
    bond_start = [b.GetBeginAtomIdx() for b in bonds]
    bond_end   = [b.GetEndAtomIdx() for b in bonds]

    # Build graph
    G = dgl.DGLGraph()
    G.add_nodes(len(atoms))
    G.add_edges(bond_start, bond_end)
    G.add_edges(bond_end, bond_start)

    #atom feats
    atom_feats, atom_targets = atoms2vec(atoms)
    G.ndata['feats'] = atom_feats.clone().detach()

    #bond_feats
    bond_feats, bond_targets = bonds2vec(bonds)
    G.edata['feats'] = bond_feats.clone().detach()

    # Get BFS node sequence to use as target sequence
    atom_seq = torch.cat(dgl.bfs_nodes_generator(G, bfs_root))
    atom_seq = [i.item() for i in atom_seq]

    # Rearrange order according to BFS
    # Generator will try to generate in this sequence
    start_atom   = torch.zeros_like(atom_feats[0]).unsqueeze(0)
    atom_feats   = torch.cat([start_atom, atom_feats[atom_seq]], 0)
    atom_targets = atom_targets[atom_seq + [len(atom_targets) - 1,]]

    # Find a subset of atoms S such that len(S) = num_nodes
    start_idx = 0# randint(0, len(atom_feats) - num_nodes)
    end_idx   = start_idx + num_nodes
    atom_idx  = torch.arange(start_idx, end_idx)
    atom_sub  = atom_seq[start_idx:end_idx]


    num_nodes = len(atom_idx)
    if num_nodes > 12:
        num_edges = ((12 // 2) * 11) + ((num_nodes - 12) * 12)
    else:
        num_edges = (num_nodes * (num_nodes - 1)) // 2

    edge_num = 0
    all_bond_targets = torch.zeros((num_edges, bond_targets.shape[-1]))
    for i in range(len(atom_idx)):
        for j in range(i):
            idx_i = atom_idx[i]
            idx_j = atom_idx[j]

            if ((idx_i > 12) and (idx_j < (idx_i - 12))):
                continue

            if idx_i == len(atom_seq):
                edge_num += 1
                continue

            s = atom_seq[idx_j]
            e = atom_seq[idx_i]

            if G.has_edge_between(s, e):
                bond_id = G.edge_id(s, e)
                all_bond_targets[edge_num] = bond_targets[bond_id]

            edge_num += 1

    atom_targets = atom_targets[start_idx:end_idx]
    atom_feats   = atom_feats[start_idx:end_idx]

    # Self loops
    G.add_edges(G.nodes(), G.nodes())

    return G, atom_idx.unsqueeze(-1), atom_feats, atom_targets, all_bond_targets
