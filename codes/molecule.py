class Molecule():
    """Molecular setup based on OpenFermion"""

    def __init__(self, mol_config:dict) -> None:

        from qiskit.circuit import QuantumCircuit, QuantumRegister
        
        from openfermion import jordan_wigner
        from openfermion.chem import MolecularData
        from openfermionpyscf import run_pyscf

        import UsefulFunctions as uf

#<<<<<<<<<<<<<<<<<<<<     Molecular Data     >>>>>>>>>>>>>>>>>>>>>>>>>>>

        self.system = 'Molecule'

        mol_name = mol_config['molecule_name']#.upper()
        self.name = mol_name

        #Generating the corresponding geometry depending on the wanted molecule
        if   mol_name == 'HE'   : atom_list = ['He']; coord_list = [(0, 0, 0)]
        elif mol_name == 'H2'   : atom_list = ['H', 'H']; coord_list = [(0, 0, 0), (0, 0, 2)]
        elif mol_name == 'HF'   : atom_list = ['H','F']; coord_list = [(0, 0, 0),(0, 0, 2.5)]
        elif mol_name == 'LiH'  : atom_list = ['Li','H']; coord_list = [(0, 0, 0),(0, 0, 2.5)]
        elif mol_name == 'H2O'  : atom_list = ['O', 'H', 'H']; coord_list = [(2.5369, -0.1550, 0.0000), (3.0739, 0.1550, 0.0000), (2.0000, 0.1550, 0.0000)] 
        elif mol_name[0] == 'H' and mol_name[1:].isdigit():
            n_hydrogen = int(mol_name[1:])
            atom_list = ['H' for i in range(n_hydrogen)]; coord_list = [(0.0, 0.0, 2.5*i) for i in range(n_hydrogen)]
        else: raise ValueError("Unknown molecule name. Please check the contents of the Molecule class.")

        #Setup of the molecular parameters
        geometry     = [[atom_list[i], coord_list[i]] for i in range(len(atom_list))]
        basis        = mol_config['basis'] 
        multiplicity = mol_config['multiplicity']
        charge       = mol_config['charge']

        #Use of Openfermion to generate the molecular data
        self.molecule = MolecularData(
                geometry=geometry,
                basis=basis,
                multiplicity=multiplicity,
                charge=charge,
                filename=f"{mol_name}_openfermion.hdf5"
                )

        # run pyscf to get Hartree-Fock or Full CI wave function from it
        run_pyscf(self.molecule,
                  run_scf=True,
                  run_mp2=False,
                  run_cisd=False,
                  run_ccsd=False,
                  run_fci=True,
                  verbose=True)

        #Setup of the active space used (frozen 1s for H2O)
        if mol_name == 'H2O':
            self.occupied_indices = range(1)
            self.active_indices = range(1,self.molecule.n_orbitals)
        else:
            self.occupied_indices = range(0)
            self.active_indices   = range(0,self.molecule.n_orbitals)

        #Setup of an attribute with the fci energy
        self.fci_energy = self.molecule.fci_energy

        #Obtaining the Hamiltonian (from Openfermion-pyscf) and converting it into the Qiskit formalism
        hamiltonian = self.molecule.get_molecular_hamiltonian(occupied_indices=self.occupied_indices,
                                                              active_indices=self.active_indices) 
        self.hamiltonian = uf.qubitop_to_qiskitpauli(jordan_wigner(hamiltonian))

        print("Number of terms in H:", len(self.hamiltonian))

        if mol_name == 'H2O':
            self.n_electrons = self.molecule.n_electrons -2
        else:
            self.n_electrons = self.molecule.n_electrons 
        self.nqubits = 2*len(self.active_indices)

        #Set up of the initial quantum state
        reg = QuantumRegister(self.nqubits)
        initial_state = QuantumCircuit(reg)
  
        if mol_name == 'HE':
            for k in range(self.nqubits - self.molecule.n_electrons, self.nqubits):
                initial_state.x(k)    
        else:
            for k in range(self.nqubits - self.molecule.n_electrons + 2*len(self.occupied_indices), self.nqubits):
                initial_state.x(k)
        self.initial_state = initial_state
