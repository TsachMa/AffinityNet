import os
import subprocess
import time
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, freeze_support

def process_file(file):
    # Define the log file
    log_file = "error.log"

    # Remove the .pdb extension if it exists
    file_no_ext = file[:-4] if file.endswith('.pdb') else file
    # Process the file with obabel and redirect stderr to the log file
    command = f"obabel {file} -O {file_no_ext}.mol2 -p 7 2>> {log_file}"
    try:
        os.system(command)
    except Exception as e:
        with open(log_file, 'a') as f:
            f.write(f"Error processing file: {file}\n")

if __name__ == '__main__':
    # Define the directory to search in
    directory = "/Users/tsachmackey/dfs/affinity_net/PDBbind"

    freeze_support()

    # Find all files ending with "_protein.pdb" recursively
    files = subprocess.check_output(["find", directory, "-type", "f", "-name", "*_protein.pdb"]).decode().splitlines()
    total_files = len(files)
    converted_files = 0
    start_time = time.time()

    # Use multiprocessing Pool to process files concurrently
    with Pool(cpu_count()) as pool:
        for _ in tqdm(pool.imap_unordered(process_file, files), total=total_files):
            converted_files += 1

    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time} seconds")
