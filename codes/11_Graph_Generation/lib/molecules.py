import ncut
import numpy as np
import torch
import time
import pickle
from rdkit import Chem


# class of atom and bond dictionaries
class Dictionary:
    """
    word2idx and idx2word are mappings from words to idx and vice versa
    word2idx is a dictionary
    idx2word is a list
    word2num_occurence compute the number of times a given word has been added to the dictionary
    idx2num_occurence do the same, but with the index of the word rather than the word itself.
    """
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.word2num_occurence = {}
        self.idx2num_occurence = []
    def add_word(self, word):
        if word not in self.word2idx:
            # dictionaries
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
            # stats
            self.idx2num_occurence.append(0)
            self.word2num_occurence[word] = 0
        # increase counters    
        self.word2num_occurence[word]+=1
        self.idx2num_occurence[  self.word2idx[word]  ] += 1
    def get_rid_of_rare_words(self, min_num_occurence):
        new_idx2word = [ word for word in self.idx2word if self.word2num_occurence[word] >= min_num_occurence  ]
        new_word2idx = { word: idx  for idx,word in enumerate(new_idx2word) }         
        new_idx2num_occurence = [ self.word2num_occurence[word] for word in new_idx2word]   
        new_word2num_occurence = { word: self.word2num_occurence[word]  for word in new_idx2word } 
        self.word2idx = new_word2idx
        self.idx2word = new_idx2word
        self.word2num_occurence = new_word2num_occurence
        self.idx2num_occurence = new_idx2num_occurence
    def show(self):
        for idx, word in enumerate(self.idx2word):
            print(idx,'\t', word,'\t number of occurences = {}'.format(self.idx2num_occurence[idx]))
    def __len__(self):
        return len(self.idx2word)


class Molecule:
    """
    A molecule object contains the following attributes:
        ; molecule.num_atom : nb of atoms, an integer (N)
        ; molecule.atom_type : pytorch tensor of size N, each element is an atom type, an integer between 0 and num_atom_type-1
        ; molecule.atom_type_pe : pytorch tensor of size N, each element is an atom type positional encoding, an integer between 0 and num_atom-1
        ; molecule.bond_type : pytorch tensor of size N x N, each element is a bond type, an integer between 0 and num_bond_type-1 
        ; molecule.bag_of_atoms : pytorch tensor of size num_atom_type, histogram of atoms in the molecule
        ; molecule.logP_SA_cycle_normalized : the chemical property to regress, a pytorch float variable
        ; molecule.smile : the smile representation of the molecule for rdkit, a string   
    """
    def __init__(self, num_atom, num_atom_type):
        self.num_atom       = num_atom
        self.atom_type      = torch.zeros( num_atom , dtype=torch.long )
        self.atom_type_pe   = torch.zeros( num_atom , dtype=torch.long )
        self.bond_type      = torch.zeros( num_atom , num_atom, dtype=torch.long )
        self.bag_of_atoms   = torch.zeros( num_atom_type, dtype=torch.long)
        self.logP_SA        = torch.zeros( 1, dtype=torch.float)
        self.logP_SA_cycle_normalized  = torch.zeros( 1, dtype=torch.float)
        self.smile  = ''
    def set_bag_of_atoms(self):
        for tp in self.atom_type:
                self.bag_of_atoms[tp.item()] += 1
    def set_atom_type_pe(self):
        histogram={}
        for idx, tp in enumerate(self.atom_type):
            tpp=tp.item()
            if tpp not in histogram:
                histogram[tpp] = 0
            else:
                histogram[tpp] += 1
            self.atom_type_pe[idx] = histogram[tpp]
    def shuffle_indexing(self):
        idx = torch.randperm(self.num_atom)
        self.atom_type = self.atom_type[idx]
        self.atom_type_pe = self.atom_type_pe[idx]
        self.bond_type = self.bond_type[idx][:,idx]
        return idx
    def __len__(self):
        return self.num_atom


# from pytorch to smile molecule 
def from_mol_to_smile(mol, remove_aromatic=False):
    if remove_aromatic==True:
        Chem.Kekulize(mol, clearAromaticFlags=True) # remove aromatic bonds
    smile = Chem.MolToSmiles(mol)
    return smile


def symbol2atom(aug_symb):
    mylist=aug_symb.split()
    atom = Chem.Atom(mylist[0])
    if '+' in mylist:
        atom.SetFormalCharge(1)
    if '-' in mylist:
        atom.SetFormalCharge(-1)
    if 'H1' in mylist:
        atom.SetNumExplicitHs(1)   
    if 'H2' in mylist:
        atom.SetNumExplicitHs(2)    
    if 'H3' in mylist:
        atom.SetNumExplicitHs(3)
    return atom
    

def from_pymol_to_smile(pymol, atom_dict, bond_dict, remove_aromatic=False):
    N = pymol.num_atom 
    mol = Chem.RWMol()
    for tp in pymol.atom_type:
        symbol = atom_dict.idx2word[ tp.item() ]
        mol.AddAtom( symbol2atom(symbol) )
    for i in range(0,N): 
        for j in range(i+1,N): 
            tp = pymol.bond_type[i,j].item()
            bond_stg = bond_dict.idx2word[tp]
            if bond_stg!='NONE':
                if bond_stg=='SINGLE':
                    mol.AddBond(i, j, Chem.rdchem.BondType.SINGLE)
                if bond_stg=='DOUBLE':
                    mol.AddBond(i, j, Chem.rdchem.BondType.DOUBLE)
                if bond_stg=='TRIPLE':
                    mol.AddBond(i, j, Chem.rdchem.BondType.TRIPLE)
                if bond_stg=='AROMATIC':
                    #print('ISSUE: MUST BE NO AROMATIC BONDS !!!!')
                    mol.AddBond(i, j, Chem.rdchem.BondType.AROMATIC)
    smile = from_mol_to_smile(mol,remove_aromatic)
    return smile


def compute_ncut(Adj, R):
    # Apply ncut
    eigen_val, eigen_vec = ncut.ncut( Adj.numpy(), R )
    # Discretize to get cluster id
    eigenvec_discrete = ncut.discretisation( eigen_vec )
    res = eigenvec_discrete.dot(np.arange(1, R + 1)) 
    # C = np.array(res-1,dtype=np.int64)
    C = torch.tensor(res-1).long()
    return C






