import copy
import random
from operator import itemgetter

import numpy as np
import rmsd
from ase import Atoms
from rdkit import Chem
from rdkit.Chem import AllChem
from xtb.ase.calculator import XTB


class BenchMolecule:
    """
    class BenchMolecule contains functionality to test
    optimization methods on the reference molecule
    """
    def __init__(self, filename: str, smiles: str, cname: str, carcass_rmsd=True):
        """
        Parameters
        ----------
        filename : str
            Filename of .xyz file from gminorgDB
        smiles : str
            SMILES representation of the molecule
        cname : str
            Compound chemical name
        carcass_rmsd : bool
            If true -> ignore hydrogens in RMSD calculations
        """
        # load reference geometry
        self.__ref_xyz = rmsd.get_coordinates(f"./gminorgDB/glob/{filename}", "xyz")

        # calculate reference energy
        ref_sys = Atoms(self.__ref_xyz[0], self.__ref_xyz[1])
        ref_sys.calc = XTB(method="GFN2-xTB")
        self.__E = ref_sys.get_potential_energy()

        # setup name attributes
        self._smiles = smiles
        self._cname = cname

        # construct the molecule for conf search
        self.__mol = Chem.MolFromSmiles(self._smiles)
        self.__mol = Chem.AddHs(self.__mol)
        Chem.AllChem.EmbedMolecule(self.__mol)

        # find dihedrals and DoF
        self.__dihedrals = BenchMolecule.__find_torsions(self.__mol)
        self._dof = len(self.__dihedrals)

        self._carcass_rmsd = carcass_rmsd

        self.__xtb_call_count = 0
        
    def __get_xtbcc(self):
        """
        Get xtb call count
        """
        return self.__xtb_call_count
        
    def __set_xtbcc(self, value):
        """
        Set xtb call count
        """
        self.__xtb_call_count = value

    def __str__(self):
        """
        String representation of the molecule
        """
        return f"Name: {self._cname}\nSMILES: {self._smiles}\nDoF: {self._dof}\
        \nEnergy: {self.__E}\nXTB call count: {self.__get_xtbcc()}\nCarcass RMSD: {self._carcass_rmsd}"

    @staticmethod
    def __find_torsions(mol):
        """
        https://th.fhi-berlin.mpg.de/meetings/dft-workshop-2017/uploads/Meeting/Tutorial3_2017_manual.pdf
        https://github.com/adrianasupady/fafoom/blob/234d2a1c45f9ceb91de5f4d425de02c563cd9178/fafoom/deg_of_freedom.py
        Find the positions of rotatable bonds in the molecule.
        """
        # get only valuable dihedrals
        pattern_tor = Chem.MolFromSmarts("[*]~[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]~[*]")
        torsion = list(mol.GetSubstructMatches(pattern_tor))

        # remove duplicate torsion definitions from a list of atom ind. tuples.
        for_remove = []
        for x in reversed(range(len(torsion))):
            for y in reversed(range(x)):
                ix1, ix2 = itemgetter(1)(torsion[x]), itemgetter(2)(torsion[x])
                iy1, iy2 = itemgetter(1)(torsion[y]), itemgetter(2)(torsion[y])
                if (ix1 == iy1 and ix2 == iy2) or (ix1 == iy2 and ix2 == iy1):
                    for_remove.append(y)
        clean_list = [v for i, v in enumerate(torsion) if i not in set(for_remove)]

        return clean_list

    @staticmethod
    def __clear_hydrogens(sym, pos1, pos2):
        """
         remove Hs from the molecule (for carcass RMSD calculation)
        """
        new_pos1 = []
        new_pos2 = []

        for s, p1, p2 in zip(sym, pos1, pos2):
            if s != "H":
                new_pos1.append(p1)
                new_pos2.append(p2)

        return new_pos1, new_pos2

    def set_carcass_rmsd_option(self, value: bool):
        """
        If true -> ignore hydrogens in RMSD calculations
        """
        self._carcass_rmsd = value

    def get_dof(self):
        return self._dof

    def get_reference_energy(self):
        return self.__E

    def get_reference_geometry(self):
        return self.__ref_xyz[0], self.__ref_xyz[1]

    def get_default_conformer(self):
        return BenchMolecule.get_conf_sym_pos(self.__mol)

    def get_random_conformer(self):
        mol = copy.deepcopy(self.__mol)
        AllChem.EmbedMolecule(mol, maxAttempts=1, randomSeed=random.randint(1, 1000000))
        return BenchMolecule.get_conf_sym_pos(mol)

    def get_n_conformers(self, n):
        temp_mol = copy.deepcopy(self.__mol)
        return temp_mol, Chem.AllChem.EmbedMultipleConfs(temp_mol, numConfs=n, randomSeed=random.randint(1, 1000000))

    def get_xtb_call_count(self):
        tmp = self.__get_xtbcc()
        self.__set_xtbcc(0)
        return tmp

    @staticmethod
    def get_conf_sym_pos(mol):
        # RDKit object to sym-pos arrays
        data = Chem.rdmolfiles.MolToXYZBlock(mol).splitlines()[2:]
        data = [x.split() for x in data]
        symbols = [x[0] for x in data]
        positions = [(float(x[1]), float(x[2]), float(x[3])) for x in data]
        return symbols, positions

    def get_metrics_by_dihedrals(self, dihedrals, xtb_counter_on=True):
        if not self._dof:
            raise Exception(f"Can't get conformer metrics if DoF = 0, molecule: {self._cname}")
        if len(dihedrals) != self._dof:
            raise Exception(f"Dihedral list ({len(dihedrals)}) doesn't match DOF({self._dof}), molecule: {self._cname}")

        return self.get_rmsd_by_dihedrals(dihedrals), self.get_energy_by_dihedrals(dihedrals, xtb_counter_on)
    
    def dihedrals_to_sym_pos(self, dihedrals):
        if not self._dof:
            raise Exception(f"Can't get conformer metrics if DoF = 0, molecule: {self._cname}")
        if len(dihedrals) != self._dof:
            raise Exception(f"Dihedral list ({len(dihedrals)}) doesn't match DOF({self._dof}), molecule: {self._cname}")

        mol = copy.deepcopy(self.__mol)

        for dihed, idx in zip(dihedrals, self.__dihedrals):
            Chem.rdMolTransforms.SetDihedralDeg(mol.GetConformer(), *idx, dihed)

        return BenchMolecule.get_conf_sym_pos(mol)
    
    def get_energy_by_dihedrals(self, dihedrals, xtb_counter_on=True):
        symbols, positions = self.dihedrals_to_sym_pos(dihedrals)
        return self.get_energy_by_xyz(symbols, positions, xtb_counter_on)

    def get_rmsd_by_dihedrals(self, dihedrals):
        _, positions = self.dihedrals_to_sym_pos(dihedrals)
        return self.get_rmsd_by_xyz(_, positions)

    def get_rmsd_by_xyz(self, _, pos):
        if self._carcass_rmsd:
            val_rmsd = rmsd.kabsch_rmsd(*BenchMolecule.__clear_hydrogens(*self.get_reference_geometry(), pos))
        else:
            val_rmsd = rmsd.kabsch_rmsd(self.get_reference_geometry()[1], pos)
        return val_rmsd

    def get_energy_by_xyz(self, sym, pos, xtb_counter_on=True):
        if xtb_counter_on:
            tmpval = self.__get_xtbcc()
            self.__set_xtbcc(tmpval+1)
        ase_sys = Atoms(sym, pos)
        ase_sys.calc = XTB(method="GFN2-xTB")
        return np.abs(ase_sys.get_potential_energy() - self.__E)
