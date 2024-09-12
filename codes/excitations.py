from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from math import pi

def apply_sq_excitations(qubits, nqubits, param_name=None, flip_qubits=False):
    '''
    Args:
        qubits (list[int]): number of the qubits [k,i] for the single qubit excitation operator U_{ki}.
        nqubits (int): number of qubits of the circuit.
        param_name (string): name of the parameter. 
    Returns:
        The circuit implementing the QEB single excitation operator, displayed as a Qiskit QuantumCircuit.
    '''
    if param_name == None:
        theta = Parameter('theta')
    else:
        theta = Parameter(param_name)

    circuit = QuantumCircuit(nqubits)

    i, k = qubits

    if flip_qubits:
        circuit.x(k)
        circuit.barrier()

    circuit.rz(pi/2., k)
    circuit.ry(-pi/2., i)
    circuit.rz(-pi/2., i)
    circuit.cx(control_qubit=k, target_qubit=i)
    circuit.ry(theta=theta, qubit=k)
    circuit.rz(-pi/2., i)
    circuit.cx(control_qubit=k, target_qubit=i)
    circuit.ry(theta=-theta, qubit=k)
    circuit.h(i)
    circuit.cx(control_qubit=k, target_qubit=i)

    return circuit

def controlled_ry(theta, circuit, control_qubits, target_qubit):
    '''
    Implements a multi-qubit controlled Ry gate, used to represent a QEB double excitation operator in a circuit.
    Args:
        theta (float): value of the parameter of the excitation.
        circuit (QuantumCircuit): initial (i.e. before this excitation) quantum circuit.
        control_qubits (list[int]): numbers of the qubits used as controls for this multi-controlled Ry gate, in the form [i,j,k].
        target_qubit (int): number of the qubit used as a target for this multi-controlled Ry gate.
    Returns:
        circuit1 (QuantumCircuit): the quantum circuit with the added multi-controlled Ry gate, displayed as a Qiskit QuantumCircuit.
    '''

    m = len(control_qubits)

    if m == 1:
        circuit.cx(control_qubit=control_qubits[0], target_qubit=target_qubit)
        circuit.ry(-0.5*theta, target_qubit)
        circuit.cx(control_qubit=control_qubits[0], target_qubit=target_qubit)
        circuit.ry(0.5*theta, target_qubit)

        return circuit
    else:
        circuit1 = controlled_ry(0.5*theta, circuit, control_qubits[1:], target_qubit)
        circuit1.cx(control_qubit=control_qubits[1], target_qubit=target_qubit)
        circuit1 = controlled_ry(-0.5*theta, circuit, control_qubits[1:], target_qubit)
        circuit1.cx(control_qubit=control_qubits[1], target_qubit=target_qubit)

        return circuit1

def apply_dq_excitations(qubits, nqubits, param_name=None):
    '''
    Args:
        qubits (list[int]): qubit's numbers [k, l, i, j] for the double qubit excitation operator U_{klij}.
        nqubits (int): number of qubits of the circuit.
        param_hame (str) : name of the parameter.
    Returns:
        circuit (QuantumCircuit): the quantum circuit with the added QEB double excitation, displayed as a Qiskit QuantumCircuit.
    '''
    if param_name == None:
        theta = Parameter('theta')
    else:
        theta = Parameter(param_name)

    l, k, j, i = qubits

    circuit = QuantumCircuit(nqubits)
    circuit.cx(control_qubit=l,target_qubit=k)
    circuit.cx(control_qubit=j,target_qubit=i)
    circuit.cx(control_qubit=l,target_qubit=j)
    circuit.x(k)
    circuit.x(i)
    circuit = controlled_ry(theta, circuit, [k,j,i], l)
    circuit.x(k)
    circuit.x(i)
    circuit.cx(control_qubit=l,target_qubit=j)
    circuit.cx(control_qubit=l,target_qubit=k)
    circuit.cx(control_qubit=j,target_qubit=i)
    
    return circuit

def apply_y_excitations(qubit, nqubits, param_name=None):
    '''
    Args:
        qubit (int): number of the qubit involved in the Y excitation.
        nqubits (int): number of qubits in the quantum circuit.
        param_name (str): name of the parameter.
    Returns:
        circuit (QuantumCircuit): circuit with the added Y excitation, displayed as a Qiskit QuantumCircuit.
    '''

    if param_name is None:
        theta = Parameter('theta')
    else:
        theta = Parameter(param_name)

    circuit = QuantumCircuit(nqubits)
    circuit.ry(theta, qubit)

    return circuit

def apply_zy_excitations(qubits, nqubits, param_name=None):
    '''
    Args:
        qubits (list[int]): numbers of the qubits involved in the ZY excitation.
        nqubits (int): number of qubits in the quantum circuit.
        param_name (str): name of the parameter.
    Returns:
        circuit (QuantumCircuit): circuit with the added ZY excitation, displayed as a Qiskit Quantum Circuit.
    '''

    if param_name is None:
        theta = Parameter('theta')
    else:
        theta = Parameter(param_name)

    circuit = QuantumCircuit(nqubits)

    circuit.cx(qubits[0], qubits[1])
    circuit.ry(theta, qubits[1])
    circuit.cx(qubits[0], qubits[1])

    return circuit 