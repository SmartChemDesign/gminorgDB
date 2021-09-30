from operator import itemgetter
from random import randrange

import ase.io
import jsonpickle
import matplotlib.pyplot as plt
import nglview
import numpy
import numpy as np
import pickle
import pyscf
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.io.trajectory import TrajectoryWriter
from ase.optimize import LBFGS
from ase.units import Ha, Bohr, Debye
from pyscf.pbc.tools.pyscf_ase import atoms_from_ase
from pyscf.prop.polarizability.uhf import polarizability, Polarizability
from rdkit import Chem
from rdkit.Chem import AllChem
from xtb.ase.calculator import XTB
from hyperopt import Trials, fmin, tpe, hp


def loadFromSMILES(SMILES):
    molecule = Chem.MolFromSmiles(SMILES)  # convert to RDKit.Mol object
    molecule = Chem.AddHs(molecule)  # add hydrogens to the carbon carcass
    AllChem.EmbedMolecule(molecule, randomSeed=randrange(1000000))
    if molecule:
        print("Success!")
    else:
        print("Fail! Check your SMILES input")

    return molecule


def find_torsions(mol):
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


def GetAseObjectByDihedrals(molecule, dihedral_values):
    dihedrals_idxs = find_torsions(molecule)
    # set the required values of the dihedrals
    for dihed, idx in zip(dihedral_values, dihedrals_idxs):
        Chem.rdMolTransforms.SetDihedralDeg(molecule.GetConformer(), *idx, dihed)
    # convert molecule from RDKit to ASE format
    # RDKit -> symbol-position arrays
    data = Chem.rdmolfiles.MolToXYZBlock(molecule).splitlines()[2:]
    data = [x.split() for x in data]
    symbols = [x[0] for x in data]
    positions = [(float(x[1]), float(x[2]), float(x[3])) for x in data]
    # symbol-position arrays -> ase system format
    return Atoms(symbols, positions)


def GlobalOptimization(molecule, iterations_count):
    dihedrals_idxs = find_torsions(molecule)
    dof_count = len(dihedrals_idxs)
    dihedrals_space = [hp.uniform(chr(ord("A") + i), -180, 180) for i in range(dof_count)]  # a search-space
    trials = Trials()  # useful class for storing the data of each iteration

    def GetEnergyByDihedrals(dihedral_values):
        ase_sys = GetAseObjectByDihedrals(molecule, dihedral_values)
        ase_sys.calc = XTB(method="GFN2-xTB")  # set energy estimation method
        return ase_sys.get_potential_energy()

    fmin(
        GetEnergyByDihedrals,  # loss function estimator
        dihedrals_space,  # search-space
        algo=tpe.suggest,  # TPE algorithm
        max_evals=iterations_count,  # max steps
        trials=trials,  # save results of each step
    )

    pickle.dump(trials, open("glob_opt.p", "wb"))


def DrawGlobalOptimizationResults():
    def get_min_loss(loss_list):
        losses = []
        for k in range(1, len(loss_list) + 1):
            losses.append(np.min(loss_list[:k]))
        return losses

    def get_mean_loss(loss_list):
        losses = []
        for k in range(1, len(loss_list) + 1):
            losses.append(np.mean(loss_list[:k]))
        return losses

    trials = pickle.load(open("glob_opt.p", "rb"))
    data = trials.losses()
    x = np.arange(len(data))
    plt.scatter(x, data, c='r')
    plt.plot(x, get_min_loss(data), color='b', label='min value')
    plt.plot(x, get_mean_loss(data), color='g', label='mean value')
    plt.legend()
    plt.xlabel("# iteration")
    plt.ylabel("Energy")
    plt.savefig("glob_opt.jpg", dpi=600)
    plt.show()


def DrawGlobalOptimizationMovie(molecule):
    trials = pickle.load(open("glob_opt.p", "rb"))
    with TrajectoryWriter("glob_opt.traj", mode="w") as out:
        for trial in trials:
            dihedrals = [x[0] for x in trial["misc"]["vals"].values()]
            tmpmol = GetAseObjectByDihedrals(molecule, dihedrals)
            out.write(tmpmol)

    show = ase.io.read('glob_opt.traj', index=':')
    nglview.show_asetraj(show)


# https://github.com/pyscf/pyscf/issues/624
class parameters():
    # holds the calculation mode and user-chosen attributes of post-HF objects
    def __init__(self):
        self.mode = 'hf'

    def show(self):
        print('------------------------')
        print('calculation-specific parameters set by the user')
        print('------------------------')
        for v in vars(self):
            print('{}:  {}'.format(v, vars(self)[v]))
        print('\n\n')


def todict(x):
    return jsonpickle.encode(x, unpicklable=False)


def init_geo(mf, atoms):
    # convert ASE structural information to PySCF information
    if atoms.pbc.any():
        cell = mf.cell.copy()
        cell.atom = atoms_from_ase(atoms)
        cell.a = atoms.cell.copy()
        cell.build()
        mf.reset(cell=cell.copy())
    else:
        mol = mf.mol.copy()
        mol.atom = atoms_from_ase(atoms)
        mol.build()
        mf.reset(mol=mol.copy())


class PYSCF(Calculator):
    # PySCF ASE calculator
    # by Jakob Kraus
    # units:  ase         -> units [eV,Angstroem,eV/Angstroem,e*A,A**3]
    #         pyscf       -> units [Ha,Bohr,Ha/Bohr,Debye,Bohr**3]

    implemented_properties = ['energy', 'forces', 'dipole', 'polarizability']

    def __init__(self, restart=None, ignore_bad_restart_file=False,
                 label='PySCF', atoms=None, directory='.', **kwargs):
        # constructor
        Calculator.__init__(self, restart, ignore_bad_restart_file,
                            label, atoms, directory, **kwargs)
        self.initialize(**kwargs)

    def initialize(self, mf=None, p=None):
        # attach the mf object to the calculator
        # add the todict functionality to enable ASE trajectories:
        # https://github.com/pyscf/pyscf/issues/624
        self.mf = mf
        self.p = p
        self.mf.todict = lambda: todict(self.mf)
        self.p.todict = lambda: todict(self.p)

    def set(self, **kwargs):
        # allow for a calculator reset
        changed_parameters = Calculator.set(self, **kwargs)
        if changed_parameters:
            self.reset()

    def get_polarizability(self, atoms=None):
        return self.get_property('polarizability', atoms)

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):

        Calculator.calculate(self, atoms=atoms, properties=properties, system_changes=system_changes)

        # update your mf object with new structural information
        init_geo(self.mf, atoms)

        # further update your mf object for post-HF methods
        if hasattr(self.mf, '_scf'):
            self.mf._scf.kernel()
            self.mf.__init__(self.mf._scf)
            for v in vars(self.p):
                if v != 'mode':
                    setattr(self.mf, v, vars(self.p)[v])
        self.mf.kernel()
        e = self.mf.e_tot

        if self.p.mode.lower() == 'ccsd(t)':
            e += self.mf.ccsd_t()

        self.results['energy'] = e * Ha

        if 'forces' in properties:
            gf = self.mf.nuc_grad_method()
            gf.verbose = self.mf.verbose
            if self.p.mode.lower() == 'dft':
                gf.grid_response = True
            forces = -1. * gf.kernel() * (Ha / Bohr)
            totalforces = []
            totalforces.extend(forces)
            totalforces = numpy.array(totalforces)
            self.results['forces'] = totalforces

        if hasattr(self.mf, '_scf'):
            self.results['dipole'] = self.mf._scf.dip_moment(verbose=self.mf._scf.verbose) * Debye
            self.results['polarizability'] = Polarizability(self.mf._scf).polarizability() * (Bohr ** 3)
        else:
            self.results['dipole'] = self.mf.dip_moment(verbose=self.mf.verbose) * Debye
            self.results['polarizability'] = Polarizability(self.mf).polarizability() * (Bohr ** 3)


def LocalOptimization(molecule):
    trials = pickle.load(open("glob_opt.p", "rb"))
    ase_sys = GetAseObjectByDihedrals(molecule, [x[0] for x in trials.best_trial["misc"]["vals"].values()])
    mol = pyscf.M(atom=atoms_from_ase(ase_sys), basis='cc-pVDZ', spin=0, charge=0)
    mf = mol.UHF()
    mf.verbose = 3
    mf.kernel()
    mf = mf.MP2()

    p = parameters()
    p.mode = 'mp2'
    p.verbose = 5
    p.show()

    mf.verbose = p.verbose
    ase_sys.calc = PYSCF(mf=mf, p=p)

    fmax = 1e-3 * (Ha / Bohr)
    dyn = LBFGS(ase_sys, logfile="loc_opt.log", trajectory="loc_opt.traj")
    dyn.run(fmax=fmax)
