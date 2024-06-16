from rdkit.Chem.rdmolfiles import MolFromPDBFile
from rdkit.Chem.rdchem import Mol
import numpy as np
from rdkit.Chem import AddHs, AssignStereochemistry, HybridizationType, ChiralType, BondStereo
from rdkit.Chem.AllChem import ComputeGasteigerCharges
import os 

def pdb_to_rdkit_mol(pdb_filepath: str): 

    #check if the file exists
    if not os.path.exists(pdb_filepath):
        raise FileNotFoundError(f"File {pdb_filepath} not found")

    mol = MolFromPDBFile(pdb_filepath, removeHs=True)

    return mol

def get_protein_or_ligand_ids(pdb_filepath: str) -> list: 
    """
    Extracts whether an atom is a protein or a ligand from a given PDB file.
    Parameters:
        pdb_filepath (str): Path to the PDB file.
    Returns:
        atom_types (list): A list of strings containing the type of each atom in the PDB file.
        it is either -1 for protein or 1 for ligand.

    Description: 
        All atoms in lines starting with "ATOM" are considered as protein atoms and all atoms 
        in lines starting with "HETATM" are considered as ligand atoms.
    """

    atom_types = []
    with open(pdb_filepath, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                atom_types.append(-1)
            elif line.startswith('HETATM'):
                atom_types.append(1)

    return atom_types

def rdkit_mol_featurizer(mol: Mol, protein_or_ligand_ids: list) -> tuple: 
    """
    Extracts graph features from a given RDKit molecule object.
    Parameters:
        mol (rdkit.Chem.rdchem.Mol): RDKit molecule object.
        protein_or_ligand_ids (list): A list of ints describing whether each atom in the molecule is a protein (-1) or a ligand (1) atom. 
    Returns:
        node_features (np.ndarray): A 2D array of shape (num_atoms, num_node_features) containing node features:
            (protein_or_ligand_id, atomic_num, atomic_mass, aromatic_indicator, ring_indicator, hybridization, chirality, num_heteroatoms, degree, num_hydrogens, partial_charge, formal_charge, num_radical_electrons)
        edge_features (np.ndarray): A 2D array of shape (num_bonds, num_edge_features) containing edge features.
        edge_indices (np.ndarray): A 2D array of shape (num_bonds, 2) containing the indices of the atoms connected by each bond.
    """

    AssignStereochemistry(mol, force=True, cleanIt=True)
    # Compute Gasteiger charges
    ComputeGasteigerCharges(mol)
    # Initialize a list to store feature vector for each node
    node_features = []

    assert len(protein_or_ligand_ids) == mol.GetNumAtoms(), "The number of atoms in the molecule and the number of atom types do not match."

    # Iterate over each atom in the molecule and calculate node features
    for atom in mol.GetAtoms():
        protein_or_ligand_id = protein_or_ligand_ids[atom.GetIdx()] 
        # Calculate node features
        atomic_num = atom.GetAtomicNum()
        atomic_mass = atom.GetMass()
        aromatic_indicator = int(atom.GetIsAromatic())
        ring_indicator = int(atom.IsInRing())
        hybridization_tag = atom.GetHybridization()
        if hybridization_tag == HybridizationType.SP:
            hybridization = 1
        elif hybridization_tag == HybridizationType.SP2:
            hybridization = 2
        elif hybridization_tag == HybridizationType.SP3:
            hybridization = 3
        else:
            hybridization = 0

        chiral_tag = atom.GetChiralTag()

        if chiral_tag == ChiralType.CHI_TETRAHEDRAL_CW:
            chirality = 1

        elif chiral_tag == ChiralType.CHI_TETRAHEDRAL_CCW:
            chirality = -1
        else:
            chirality = 0
        num_heteroatoms = len([bond for bond in atom.GetBonds() if bond.GetOtherAtom(atom).GetAtomicNum() != atom.GetAtomicNum()])
        degree = atom.GetDegree()
        num_hydrogens = atom.GetTotalNumHs()
        partial_charge = atom.GetProp('_GasteigerCharge')
        formal_charge = atom.GetFormalCharge()
        num_radical_electrons = atom.GetNumRadicalElectrons()
        # Append node features to list
        node_features.append((protein_or_ligand_id, atomic_num, atomic_mass, aromatic_indicator, ring_indicator, hybridization,
        chirality, num_heteroatoms, degree, num_hydrogens, partial_charge, formal_charge, num_radical_electrons))

    # Initialize a list to store edge features and indices
    edge_indices, edge_features = [], []
    # Iterate over each bond in the molecule and calculate edge features
    for bond in mol.GetBonds():
        # Get edge indices
        idx1, idx2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        # Calculate edge features
        bond_order = bond.GetBondTypeAsDouble()
        aromaticity = int(bond.GetIsAromatic())
        conjugation = int(bond.GetIsConjugated())
        ring = int(bond.IsInRing())
        stereochemistry_tag = bond.GetStereo()
        if stereochemistry_tag == BondStereo.STEREOZ:
            stereochemistry = 1
        elif stereochemistry_tag == BondStereo.STEREOE:
            stereochemistry = -1
        else:
            stereochemistry = 0
        # Append edge indices to list, duplicating to account for both directions
        edge_indices.append((idx1, idx2))
        edge_indices.append((idx2, idx1))
        # Append edge features to list, duplicating to account for both directions
        edge_features.append((bond_order, aromaticity, conjugation, ring, stereochemistry))
        edge_features.append((bond_order, aromaticity, conjugation, ring, stereochemistry))


    return np.array(node_features, dtype='float64'), np.array(edge_features, dtype='float64'), np.array(edge_indices, dtype='int64')