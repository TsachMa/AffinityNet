import openbabel

def remove_water_molecules(input_file, output_file):
    """
    Args: 
        input_file (str): The path to the input file (mol2)
        output_file (str): The path to the output file (mol2)
    """
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("mol2", "mol2")

    mol = openbabel.OBMol()
    obConversion.ReadFile(mol, input_file)

    water_residue_names = ["HOH", "H2O", "TIP3"]  # Add more if needed

    waters_to_remove = []
    for residue in openbabel.OBResidueIter(mol):
        if residue.GetName() in water_residue_names:
            waters_to_remove.append(residue)

    for water in waters_to_remove:
        mol.DeleteResidue(water)

    obConversion.WriteFile(mol, output_file)
