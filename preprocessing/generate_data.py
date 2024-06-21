import sys 
sys.path.append('../')  

import os
import argparse

from src.data.parse_data import create_cleaned_dataset
from src.data.add_H_and_mol2_chimera import add_H_and_convert_to_mol2

def main(): 
    #take as an argument the path to your data
    parser = argparse.ArgumentParser(description='Create cleaned dataset')
    parser.add_argument('--PDB_data_path', type=str, help='path to PDBbind dataset')

    args = parser.parse_args()

    PDBbind_data_path = os.path.join(args.PDB_data_path, 'PDBbind_2020_data.csv')
    general_set_PDBs_path = os.path.join(args.PDB_data_path, 'v2020-other-PL')
    refined_set_PDBs_path = os.path.join(args.PDB_data_path, 'refined-set')
    output_name = os.path.join(args.PDB_data_path, 'cleaned_dataset.csv')

    create_cleaned_dataset(PDBbind_data_path, general_set_PDBs_path, refined_set_PDBs_path, output_name, plot = True)
    add_H_and_convert_to_mol2(general_set_PDBs_path, pocket_files=False, cli=True)
    add_H_and_convert_to_mol2(refined_set_PDBs_path, pocket_files=False, cli=True)

if __name__ == '__main__':
    main()
