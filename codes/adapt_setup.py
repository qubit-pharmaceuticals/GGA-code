class AdaptSetup():
    """Setup config for ADAPT-VQE."""

    def __init__(self, adapt_config:dict, nqubits:int, n_electrons:int = 0) -> None:

        import numpy as np

        from qiskit_algorithms.optimizers import L_BFGS_B, COBYLA, POWELL

        import OperatorPool as op

        for key in adapt_config:
            setattr(self, key, adapt_config[key])

#<<<<<<<<<<<<<<<<<<<<<<<<<<     ADAPT     >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        if self.shots == 0:
            self.shots = None

        if self.algorithm == 'adapt':
            if self.optimizer == 'COBYLA':
                self.optimizer = COBYLA()
            elif self.optimizer == 'BFGS':
                self.optimizer = L_BFGS_B()
            elif self.optimizer == 'POWELL':
                self.optimizer = POWELL()
        elif self.algorithm == 'gga':
            if self.optimizer == 'BFGS':
                self.optimizer = 'L-BFGS-B'

#<<<<<<<<<<<<<<<<     Ansatz Parameters     >>>>>>>>>>>>>>>>>>>>>>>>>>>

        if self.pool_name == 'SD':
            self.pool = op.SD_qubit_excitation_pool(nqubits, n_electrons)
        elif self.pool_name == 'GSD':
            self.pool = op.GSD_qubit_excitation_pool(nqubits)
        elif self.pool_name == 'fermionic':
            self.pool = op.SD_fermionic_excitation_pool(n_ubits, n_electrons)
        elif self.pool_name == 'ising_minimal':
            self.pool = op.Ising_minimal_pool(nqubits)

        self.generators = self.pool.generators
        self.excitations = self.pool.operators

        self.verbose = True
        
        return