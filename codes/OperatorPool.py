from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit_nature.second_q.circuit.library.ansatzes.utils import generate_fermionic_excitations

import UsefulFunctions as uf
import excitations as ex

class SD_qubit_excitation_pool:
    '''
    Representations of the Qubit Excitations-Based (QEB) pool of operators, containing the single and double excitation operators.
    Possesses both the:
        circuits of each excitation, displayed as a Qiskit QuantumCircuit unit
        Pauli strings of each excitation, displayed as a Qiskit SparsePauliOp unit
    '''

    def __init__(self, qubits, n_electrons):

        self.single_excitations: list[QuantumCircuit] = []
        self.double_excitations: list[QuantumCircuit] = []
        self.single_generators: list[SparsePauliOp] = []
        self.double_generators: list[SparsePauliOp] = []
        self.names: list[str] = []
        self.acting_on: list[str] = []

        self.qubits = qubits
        self.n_spin_orb = self.qubits
        self.n_electrons = n_electrons

        count = 0
        Xs, Ys, Zs = uf.create_pauli_operators(self.qubits)

        #Single excitation operators
        for k in range(self.n_electrons):
            for i in range(self.n_electrons, self.n_spin_orb):
                spin_check = k + i
                if spin_check %2 == 0:
                    iprime, kprime = qubits-1-i, qubits-1-k

                    param_name = 'phi_' + str(count)
                    self.single_excitations.append(ex.apply_sq_excitations(qubits=[iprime,kprime], nqubits=self.qubits, param_name=param_name, flip_qubits=False))
                    single_generator = SparsePauliOp(Xs[iprime] & Ys[kprime], 0.5) - SparsePauliOp(Ys[iprime] & Xs[kprime], 0.5)
                    self.single_generators.append(single_generator)

                    count += 1
                    self.names.append(str(k)+'_'+str(i))
                    self.acting_on.append([kprime,iprime])

        print('n electrons: ', self.n_electrons)
        print('n spin orb: ', self.n_spin_orb)

        #Double excitation operators
        for I in range(self.n_electrons-1):
            for J in range(I + 1, self.n_electrons):
                for K in range(self.n_electrons, self.n_spin_orb-1):
                    for L in range(K + 1, self.n_spin_orb):
                        qubit_list = [I,J,K,L]
                        for q in range(len(qubit_list)):
                            qubit_list[q] = qubits-1-qubit_list[q]
                        i,j,k,l = qubit_list
                        spin_check = i + j + k + l
                        if spin_check%2 == 0:
                            bot_spin_check = k + l 

                            if bot_spin_check%2 == 0:
                                if k%2 == 0 and l%2 == 0 and j%2 == 0 and i%2 == 0:
                                    pass
                                elif k%2 == 1 and l%2 == 1 and j%2 == 1 and i%2 == 1:
                                    pass
                                else:
                                    break
                            eq20yordanov = [i,j,k,l] 
                            param_name = 'phi_' + str(count)
                            self.double_excitations.append(ex.apply_dq_excitations(qubits=eq20yordanov, nqubits=qubits, param_name=param_name))
                            double_generator = SparsePauliOp(Xs[i] & Ys[j] & Xs[k] & Xs[l], .125)
                            double_generator = double_generator + SparsePauliOp(Ys[i] & Xs[j] & Xs[k] & Xs[l], .125)
                            double_generator = double_generator + SparsePauliOp(Ys[i] & Ys[j] & Ys[k] & Xs[l], .125)
                            double_generator = double_generator + SparsePauliOp(Ys[i] & Ys[j] & Xs[k] & Ys[l], .125)
                            double_generator = double_generator + SparsePauliOp(Xs[i] & Xs[j] & Ys[k] & Xs[l], -.125)
                            double_generator = double_generator + SparsePauliOp(Xs[i] & Xs[j] & Xs[k] & Ys[l], -.125)
                            double_generator = double_generator + SparsePauliOp(Ys[i] & Xs[j] & Ys[k] & Ys[l], -.125)
                            double_generator = double_generator + SparsePauliOp(Xs[i] & Ys[j] & Ys[k] & Ys[l], -.125)
                            self.double_generators.append(double_generator)

                            count += 1
                            self.names.append(str(I) + '_' + str(J) + '_' + str(K) + '_' + str(L))
                            self.acting_on.append(qubit_list)

        self.generators = self.single_generators + self.double_generators
        self.operators = self.single_excitations + self.double_excitations

        print('Length of single excitations operators list: ', len(self.single_excitations))
        print('Length of double excitations operators list: ', len(self.double_excitations))
        print('Excitations', self.names)

class Ising_minimal_pool:
    '''
    This class represents the specific minimal pool for the 1D transverse-field Ising model.
    Possesses both for each excitation the:
        circuit, displayed as a Qiskit QuantumCircuit unit.
        Pauli string, displayed as a Qiskit SparsePauliOp unit.
    '''

    def __init__(self, qubits):

        self.qubits = qubits

        self.y_operators: list[QuantumCircuit] = []
        self.y_generators: list[SparsePauliOp] = []
        self.zy_operators: list[QuantumCircuit] = []
        self.zy_generators: lisŧ[SparsePauliOp] = []

        Xs, Ys, Zs = uf.create_pauli_operators(self.qubits)

        count = 0

        #Y excitation operators
        for n in range(self.qubits):
            param_name = 'phi_' + str(count)

            self.y_operators.append(ex.apply_y_excitations(n, self.qubits, param_name))
            y_generator = SparsePauliOp(Ys[n], 1.0)
            self.y_generators.append(y_generator)

            count += 1
        
        #ZY excitation operators
        for n in range(self.qubits -1):
            param_name = 'phi_' + str(count)
            
            self.zy_operators.append(ex.apply_zy_excitations([n,n+1], self.qubits, param_name))
            zy_generator = SparsePauliOp(Zs[n] & Ys[n+1], 1.0)
            self.zy_generators.append(zy_generator)

            count += 1

        self.generators = self.y_generators + self.zy_generators
        self.operators = self.y_operators + self.zy_operators
        
        print('Number of Y excitation operators: ', len(self.y_operators))
        print('Number of ZY excitation operators: ', len(self.zy_operators))
