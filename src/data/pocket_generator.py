from openbabel import openbabel
from openbabel import pybel

def read_molecule(filename):
    """Read a molecule from a file."""
    return next(pybel.readfile(filename.split('.')[-1], filename))

def find_pocket_atoms(protein, ligand, radius):
    """Find protein atoms within a certain radius of the ligand."""
    pocket_atoms = []
    for protein_atom in protein:
        for ligand_atom in ligand:
            distance = protein_atom.OBAtom.GetDistance(ligand_atom.OBAtom)
            if distance <= radius:
                pocket_atoms.append(protein_atom)
                break
    return pocket_atoms

def save_pocket_atoms(pocket_atoms, output_filename):
    """Save the pocket atoms to a PDB file."""
    mol = pybel.Molecule(openbabel.OBMol())
    for atom in pocket_atoms:
        mol.OBMol.AddAtom(atom.OBAtom)
    mol.write("pdb", output_filename, overwrite=True)

def find_and_save_pocket_atoms(protein_filename, ligand_filename, radius, output_filename):
    """Find and save protein atoms within a certain radius of the ligand.
    
    Args:   
        protein_filename (str): The filename of the protein PDB file.
        ligand_filename (str): The filename of the ligand PDB file.
        radius (float): The radius in angstroms.
        output_filename (str): The filename of the output PDB file
        
    Returns:
        None
    """

    protein = read_molecule(protein_filename)
    ligand = read_molecule(ligand_filename)
    pocket_atoms = find_pocket_atoms(protein, ligand, radius)
    save_pocket_atoms(pocket_atoms, output_filename)
    print(f"Pocket atoms saved to {output_filename}")



