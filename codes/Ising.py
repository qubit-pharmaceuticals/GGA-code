class IsingModel():
    "Ising model settings"

    def __init__(self, ising_config: dict) -> None:

        from qiskit.circuit import QuantumCircuit, QuantumRegister

        import UsefulFunctions as uf

        self.system = 'Ising'
        self.name = ising_config['ising']

        if ising_config['ising'][5:].isdigit():
            self.nqubits = int(ising_config['ising'][5:])
        else:
            raise ValueError

        self.h = ising_config['field']
        self.J = ising_config['coupling']

        self.hamiltonian = uf.Ising_Hamiltonian(self.nqubits, self.J, self.h)

        reg = QuantumRegister(self.nqubits)
        initial_state = QuantumCircuit(reg)

        for i in range(self.nqubits):
            initial_state.x(i)
            initial_state.h(i)

        self.initial_state = initial_state
