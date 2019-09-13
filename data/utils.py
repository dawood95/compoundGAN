import dgl
import torch

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
        except: idx = len(Library.atom_list)
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
        idx = torch.Tensor([atom.GetIsAromatic()]).long().unsqueeze(0)
        aromatic_emb.append(idx)

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
        bs = bond.GetStereo()

        emb = [bt == Chem.rdchem.BondType.SINGLE,
               bt == Chem.rdchem.BondType.DOUBLE,
               bt == Chem.rdchem.BondType.TRIPLE,
               bt == Chem.rdchem.BondType.AROMATIC]
        emb = torch.Tensor(emb).long()
        bond_emb.append(emb)

        emb = torch.Tensor([bond.GetIsConjugated()]).long()
        conjugated.append(emb)

        emb = torch.Tensor([bond.IsInRing()]).long()
        ring.append(emb)

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
    feats = torch.cat(feats, 0)
    G.ndata['feats'] = feats.float()

    #bond_emb, conjugated, ring, chirality
    feats = []
    bond_feats = bonds2vec(bonds)
    for i in range(len(bonds)):
        f = [bf[i] for bf in bond_feats]
        f = torch.cat(f, dim=0)
        feats.append(f.unsqueeze(0))
    feats = torch.cat(feats, 0)
    G.edata['feats'] = feats.float()

    bfs_edge = torch.cat(dgl.bfs_edges_generator(G, bfs_root))
    atom_order = []
    bond_order = []
    for edge in bfs_edge:
        s = bonds[edge].GetBeginAtomIdx()
        e = bonds[edge].GetEndAtomIdx()
        if s not in atom_order: atom_order.append(s)
        if e not in atom_order: atom_order.append(e)
        bond_order.append(edge.item())

    atom_feats = [torch.cat(af, dim=0)[atom_order] for af in atom_feats]
    bond_feats = [torch.cat(bf, dim=0)[bond_order] for bf in bond_feats]

    return G, atom_feats, bond_feats
