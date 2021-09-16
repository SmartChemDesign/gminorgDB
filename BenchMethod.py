import joblib
import os
import time
from copy import deepcopy


class BenchMethod:
    def __init__(self, method=None, max_iters=None,
                 verbose=True, molecule_set=None, **method_params):
        self.__method = method
        self._method_name = self.__method.__name__
        self._max_iters = max_iters
        self._method_params = method_params
        self.__verbose = verbose
        self.__mol_list = deepcopy(molecule_set)

        self.__load()

        if not os.path.exists(self._method_name):
            os.mkdir(f"./{self._method_name}")

        self.__dump(f"initialized {self._method_name} benchmark")

    def __dump(self, message):
        with open(f"./{self._method_name}/dump_{self._method_name}.log", "a+") as f:
            print(f"{time.strftime('%H:%M:%S', time.localtime())}", file=f)
            print(message, file=f)
        if self.__verbose:
            print(message)

    def __load(self):
        try:
            self.__data = joblib.load(f"./{self._method_name}/{self._method_name}.p")
        except FileNotFoundError:
            self.__data = []

    def __save(self):
        if len(self.__data) != 0:
            joblib.dump(self.__data, f"./{self._method_name}/{self._method_name}.p", compress=("gzip", 3))

    def __calc_molecule(self, molecule):
        self.__dump(f"calc molecule: {molecule._cname}, dof = {molecule.get_dof()}")
        iterations = self.__method(molecule, self._max_iters, **self._method_params)
        return {"molecule_name": molecule._cname,
                "smiles": molecule._smiles,
                "molecule_dof": molecule._dof,
                # "molecule_structure": molecule.get_reference_geometry(),
                # "molecule_energy": molecule.get_reference_energy(),
                "iterations": iterations,
                }

    def run(self):
        self.__dump(f"===START===")
        try:
            computed_mols = [m["molecule_name"] for m in self.__data]
        except IndexError:
            computed_mols = []

        self.__dump(f"already calculated: {len(computed_mols)} / {len(self.__mol_list)}")

        for mol in self.__mol_list:
            if mol._cname not in computed_mols:
                mol.get_xtb_call_count()
                self.__data.append(self.__calc_molecule(mol))
                self.__save()

        self.__dump(f"===DONE===")

    def purge_data(self):
        self.__data.clear()
        os.system(f"rm ./{self._method_name}/{self._method_name}.p")
        os.system(f"rm ./{self._method_name}/dump_{self._method_name}.log")
