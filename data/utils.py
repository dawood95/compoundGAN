import dgl
import torch
import numpy as np

from torch.nn import functional as F
from rdkit import Chem

class Library:
    atom_list = ['C', 'N', 'O', 'S', 'F', 'Si', 'P',\
                 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe',\
                 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl',\
                 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', \
                 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu',\
                 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr',\
                 'Cr', 'Pt', 'Hg', 'Pb']
    charge_list = [-3, -2, -1, 0, 1, 2, 3]
    electron_list = [0, 1, 2]
    chirality_list = ['R', 'S']

def atoms2vec(atoms):
    atom_emb = []
    charge_emb = []
    electron_emb = []
    chirality_emb = []
    aromatic_emb = []
    for atom in atoms:
        # Element
        try: idx = Library.atom_list.index(atom.GetSymbol())
        except: raise ValueError
        idx = torch.Tensor([idx]).long()
        emb = F.one_hot(idx, len(Library.atom_list)+1)
        atom_emb.append(emb)

        # Charge
        idx = torch.Tensor([atom.GetFormalCharge()+3]).long()
        emb = F.one_hot(idx, len(Library.charge_list))
        charge_emb.append(emb)

        # Radical Electrons
        idx = torch.Tensor([atom.GetNumRadicalElectrons()]).long()
        emb = F.one_hot(idx, len(Library.electron_list))
        electron_emb.append(emb)

        # Aromatic
        idx = torch.Tensor([atom.GetIsAromatic()]).long()
        emb = F.one_hot(idx, 2)
        aromatic_emb.append(emb)

        # Chirality
        try: idx = Library.chirality_list.index(atom.GetProp('_CIPCode'))
        except: idx = len(Library.chirality_list)
        idx = torch.Tensor([idx]).long()
        emb = F.one_hot(idx, len(Library.chirality_list)+1)
        chirality_emb.append(emb)

    return atom_emb, charge_emb, electron_emb, chirality_emb, aromatic_emb

def bonds2vec(bonds):
    conjugated = []
    ring = []
    bond_emb = []
    chirality = []
    for bond in bonds:
        bt = bond.GetBondType()
        bs = str(bond.GetStereo())

        emb = [False,
               bt == Chem.rdchem.BondType.SINGLE,
               bt == Chem.rdchem.BondType.DOUBLE,
               bt == Chem.rdchem.BondType.TRIPLE,
               bt == Chem.rdchem.BondType.AROMATIC]
        emb = torch.Tensor(emb).long()
        bond_emb.append(emb)

        emb = torch.Tensor([bond.GetIsConjugated()]).long()
        conjugated.append(torch.Tensor([emb == 0, emb == 1]).long())

        emb = torch.Tensor([bond.IsInRing()]).long()
        ring.append(torch.Tensor([emb == 0, emb == 1]).long())

        emb = [bs=="STEREONONE",
               bs=="STEREOANY",
               bs=="STEREOZ",
               bs=="STEREOE"]
        emb = torch.Tensor(emb).long()
        chirality.append(emb)

    return bond_emb, conjugated, ring, chirality

def mol2graph(mol):
    G = dgl.DGLGraph()
    bfs_root = list(Chem.CanonicalRankAtoms(mol)).index(0)

    mol = Chem.AddHs(mol)
    atoms = list(mol.GetAtoms())
    bonds = list(mol.GetBonds())

    G.add_nodes(len(atoms))

    edge_start = [b.GetBeginAtomIdx() for b in bonds]
    edge_end   = [b.GetEndAtomIdx() for b in bonds]
    G.add_edges(edge_start, edge_end)
    G.add_edges(edge_end, edge_start)

    #atoms_emb, charge_emb, electron_emb, chirality_emb, aromatic_emb
    feats = []
    atom_feats = atoms2vec(atoms)
    for i in range(len(atoms)):
        f = [af[i] for af in atom_feats]
        f = torch.cat(f, dim=1)
        feats.append(f)
    G.ndata['feats'] = torch.cat(feats, 0).float()

    #bond_emb, conjugated, ring, chirality
    feats = []
    bond_feats = bonds2vec(bonds)
    for i in range(len(bonds)):
        f = [bf[i] for bf in bond_feats]
        f = torch.cat(f, dim=0)
        feats.append(f.unsqueeze(0))
    G.edata['feats'] = torch.cat(feats * 2, 0).float()

    # Get BFS node sequence to use as target sequence
    atom_seq = torch.cat(dgl.bfs_nodes_generator(G, bfs_root))
    atom_seq = [i.item() for i in atom_seq]

    # Rearrange order according to BFS
    # Generator will try to generate in this sequence
    target = []
    for i in atom_seq:
        _target = []
        for af in atom_feats:
            _target.append(af[i].nonzero()[0, 1].item())
        target.append(_target)
    target.append([len(Library.atom_list), 3, 0, 2, 0])

    bond_target = []
    for i in range(len(atom_seq)):
        _bond_target = []
        for j in range(i):
            s = atom_seq[j]
            e = atom_seq[i]
            if G.has_edge_between(s, e):
                ## could be the second set of bonds too, thus modulo
                bond_id = G.edge_id(s, e)%len(bonds)
                _target = [bf[bond_id].nonzero()[0] for bf in bond_feats]
                _target = [x.item() for x in _target]
                _bond_target.append(_target)
            else:
                _bond_target.append([0, 0, 0, 0])
        bond_target.append(_bond_target)
    bond_target.append([[0, 0, 0, 0] for _ in range(len(bond_target[-1])+1)])

    G.add_edges(G.nodes(), G.nodes())

    target = np.array(target)
    bond_target = np.array(bond_target)
    
    return G, target, bond_target
