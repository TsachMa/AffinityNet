import pandas as pd
import numpy as np
import seaborn as sns
import os

def create_cleaned_dataset(PDBbind_dataset_path, general_set_PDBs_path, refined_set_PDBs_path, output_name, plot = False):
  """
  Produces a csv file containing PDB id, binding affinity, and set (general/refined)
  
  Inputs:
  1) PDBbind_dataset_path: path to PDBbind dataset; dataset is included in github repository as 'PDBbind_2020_data.csv'
  2) general_set_PDBs_path: path to PDBbind general set excluding refined set PDBs
  3) refined_set_PDBs_path: path to PDBbind refined set PDBs
  4) output_name: name for the output csv file. Must end in .csv
  5) plot = True will generate a plot of density as a function of binding affinity for general
     and refined sets
     
  Output:
  1) A cleaned csv containing PDB id, binding affinity, and set (general/refined):
     'output_name.csv'
  """
  
  # load dataset
  data = pd.read_csv(PDBbind_dataset_path)
  
  # check for NaNs in affinity data
  if data['-log(Kd/Ki)'].isnull().any() != False:
    print('There are NaNs present in affinity data!')
    
  # create list of PDB id's
  pdbid_list = list(data['PDB ID'])
  data['set'] = ""
  # remove affinity values that do not have structural data by searching PDBs
  missing = []
  for i in range(len(pdbid_list)):
    pdb = pdbid_list[i]
    if os.path.isdir(os.path.join(str(general_set_PDBs_path), pdb)):
        data.loc[i, 'set'] = 'general'
    elif os.path.isdir(os.path.join(str(refined_set_PDBs_path), pdb)):
        data.loc[i, 'set'] = 'refined'
    else:
        missing.append(pdb)

  data = data[~np.in1d(data['PDB ID'], list(missing))]

  # write out csv of cleaned dataset
  data[['PDB ID', '-log(Kd/Ki)', 'set']].to_csv(output_name, index=False)
  
  if plot == True:
    # plot affinity distributions for general and refined sets
    grid = sns.FacetGrid(data, row='set', row_order=['general', 'refined'], aspect=2)
    grid.map(sns.distplot, '-log(Kd/Ki)')
    #save the plot 
    grid.savefig('affinity_distributions.png')
  else:
    return
