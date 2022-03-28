import time

import numpy as np
import pandas as pd
from rdkit import Chem
from tqdm import tqdm

from BenchDrawer import BenchDrawer
from BenchMethod import BenchMethod
from BenchMolecule import BenchMolecule

# import first 7 database instances for benchmarking example
df = pd.read_csv("../gminorgDB/globdata.csv", sep=';')
mol_list = []
for index, row in df[:7].iterrows():
    fname = row.filename
    smiles = row["smiles"]
    cname = row["name"]
    #                 filename  SMILES   compound name
    bm = BenchMolecule(fname,   smiles,     cname) # this object contains info about the database instance
    if bm.get_dof() > 0:
        mol_list.append(bm) 


# benchmarking function
def dg_exte_test(mol, iternum):
    rmsd_list = []
    E_list = []
    time_list = []
    xtbcalls_list = []

    confmol, ids = mol.get_n_conformers(iternum) # <- function to benchmark. In this example, RDKit's built-in ETKDG method 
    for _id in tqdm(ids):
        xtbcalls_list.append(_id + 1)

        start_t = time.time()
        
        # convert to XYZ format, for the energy calculation
        data = Chem.rdmolfiles.MolToXYZBlock(confmol, confId=_id).splitlines()[2:]
        data = [x.split() for x in data]
        sym = [x[0] for x in data]
        pos = [(float(x[1]), float(x[2]), float(x[3])) for x in data]

        E_list.append(mol.get_energy_by_xyz(sym, pos)) # <- energy estimator function
        rmsd_list.append(mol.get_rmsd_by_xyz(sym, pos))

        end_t = time.time()
        time_list.append(end_t - start_t)

    return np.asarray(rmsd_list), np.asarray(E_list), time_list, xtbcalls_list


# init benchmark system
dg_bm = BenchMethod(dg_exte_test, max_iters=50, molecule_set=mol_list)
dg_bm.purge_data()
dg_bm.run()

# init drawing system
bd = BenchDrawer(dg_exte_test)
bd.draw_all_molecules(save=False, normalized=False)
bd.draw_summary(save=False, normalized=True)
for k, v in bd.get_method_statistics(normalized=False, with_idxs=False).items():
    print(f"{k}: mean = {v[0]}, stdev = {v[1]}, min = {v[2]}, max = {v[3]}")
