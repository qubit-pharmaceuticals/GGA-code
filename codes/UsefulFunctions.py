import os
import ast
import numpy as np
from copy import copy

from collections import deque
from scipy.optimize import minimize, Bounds

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from qiskit.primitives import Estimator, StatevectorSampler
from qiskit.quantum_info import PauliList, Pauli, SparsePauliOp

from openfermion.utils.operator_utils import count_qubits

##### Pure Qiskit functionalities #####
def expectation_value(hamiltonian, circuit, theta, shots):
    '''
    Generic function that performs the computation of the expectation value of a Hamiltonian with respect to a quantum circuit.
    Args:
        hamiltonian (SparsePauliOp): a Hamiltonian displayed in the Qiskit SparsePauliOp unit.
        circuit (QuantumCircuit): parameterised circuit with respect to the expectation value is computed.
        theta (list[float]): values of the parameters of the circuit.
        shots (int): number of shots of the computation.
    Returns:
        mean_value (float): numerical value of the expectation value of the Hamiltonian.
    '''

    estimator = Estimator()
    job = estimator.run(circuit, hamiltonian, theta, shots=shots)

    result = job.result()
    mean_value = result.values[0].real

    return mean_value

def measure_circuits(circuits, theta, shots):
    '''
    Generic function to compute the quasi-probability of each existing state for several quantum circuits, with respect to a finite number of shots.
    Args:
        circuits (list[QuantumCircuit]): list of several parameterised quantum circuits.
        theta (list[float]): list of the values of the parameters.
        shots (int): number of shots of the computation.
    Returns:
        results (list[dict]): results of the measured states with their respective probability for each circuit. 
    '''

    results = []

    for qc in circuits:
        alpha = ClassicalRegister(qc.num_qubits, "alpha")
        circuit = QuantumCircuit(QuantumRegister(qc.num_qubits), alpha)
        circuit.append(qc, range(qc.num_qubits))
        
        if len(theta) == 1:
            circuit.assign_parameters([])
        else:
            new_theta = theta[:-1]
            circuit.assign_parameters(new_theta)
        
        circuit.measure([q for q in range(circuit.num_qubits)], alpha)
        pub = (circuit, theta[:-1])

        job = StatevectorSampler().run([pub], shots=shots)
        
        result = job.result()[0].data.alpha.get_counts()
        results.append(result)

    return results

##### SDK related functions #####
def create_pauli_operator(num_qubits, qubits):
    '''
    Creates one Pauli string representing a specific operator in the pool.
    Args:
        num_qubits (int): number of qubits in the operator.
        qubits (list[int]): list of the qubits non-trivially involved in the operator.
    Returns:
        string (str): Corresponding Pauli string.
    '''
    
    elements = len(qubits)

    string = ''
    for i in range(num_qubits):
        string = string + 'I'
    
    for i in range(elements):
        j = qubits[i][1]
        if qubits[i][0] == 'X':
            string = string[:j] + 'X' + string[j+1:]
        elif qubits[i][0] == 'Y':
            string = string[:j] + 'Y' + string[j+1:]
        elif qubits[i][0] == 'Z':
            string = string[:j] + 'Z' + string[j+1:]
    
    return string

def create_pauli_operators(nqubits, flip = False):
    '''
    Creates the Pauli strings representing the operators in a specific pool.
    Args:
        nqubits (integer): number of qubits in the operator.
        flip (bool): if True, flip the orders of each single Pauli operators in each string.
    Returns: 
        3 list of Qiskir PauliList operators
    '''
    xs, ys, zs = create_pauli_strings(nqubits)
    if flip:
        xs.reverse()
        ys.reverse()
        zs.reverse()

    return PauliList(xs), PauliList(ys), PauliList(zs)

def create_pauli_strings(nqubits):
    '''
    Args:
        nqubits (int): number of qubits.
    Returns: 
        3 list of Pauli strings
    '''
    identity = ''
    for _ in range(nqubits):
        identity = identity + 'I'
    list_x_operators = create_list_p_operators(identity, 'X')
    list_y_operators = create_list_p_operators(identity, 'Y')
    list_z_operators = create_list_p_operators(identity, 'Z')

    return list_x_operators, list_y_operators, list_z_operators

def create_list_p_operators(identity, p):
    '''
    Args:
        identity (str): string only made of identity operator.
        p (str): string corresponding to one non-trivial Pauli operator.
    Returns:
        list_p_operators (list(str)): list of strings with one non-trivial Pauli operator at different spots in the Pauli string.
    '''
    list_p_operators = []
    for k in range(len(identity)):
        new_op = copy(identity)
        new_op = new_op[:k] + p + new_op[k+1:]
        list_p_operators.append(new_op)
    list_p_operators.reverse()

    return list_p_operators

##### Hamiltonian related functions ######
def qubitop_to_qiskitpauli(qubit_operator):
    '''
    Function that performs the transformation of an openfermion qubit operator into a Qiskit Pauli operator
    Args:
        qubit_operator (QubitOperator): Openfermion representation of the Hamiltonian in a qubit basis.
    Returns:
        op is a Qiskit Pauli operator, usually representing a molecular hamiltonian in the Qiskit-qubit formalism
    '''
    op = 0

    for qubit_terms, qubit_coeff in qubit_operator.terms.items():
        string_term = "I"*count_qubits(qubit_operator)
        for i, (term_qubit, term_pauli) in enumerate(qubit_terms):
            string_term = (string_term[:term_qubit] + term_pauli + string_term[term_qubit + 1 :])

        op += SparsePauliOp(Pauli(string_term), coeffs=qubit_coeff)

    return op

def Ising_Hamiltonian(qubits, coupling, field):
    '''
    Function that creates the Hamiltonian for an 1D transverse-field Ising model with open boundary conditions.
    Args:
        qubits (int): number of qubits.
        coupling (float): numerical value of the coupling constant between two qubits.
        field (float): numerical value of the field constant.
    Returns: 
        op (SparsePauliOp): Qiskit representation of the 1D transverse-field Ising Hamiltonian.
    '''

    num_terms = 0
    op = 0

    Xs, YS, Zs = create_pauli_operators(qubits)

    for n in range(qubits):
        num_terms += 1
        op += SparsePauliOp(Xs[n], field)
    
    for n in range(qubits -1):
        num_terms += 1
        op += SparsePauliOp(Zs[n] & Zs[n+1], coupling)
    
    print('Number of terms in H: ', num_terms)

    return op

###### Molecular ADAPT/GGA functions #####
def compute_minimum_in_direction(data_points, optimizer):
    '''
    Function that performs the GGA 1D local optimization.
    Args:
        data_points (list[float]): list of the measured energies for the GGA method.
        optimizer (str): name of the optimizer.
    Returns:
        output.fun is the value of the minimum extracted from the 1D local optimization
        output.x is the value of the parameter extracted from the 1D local optimization
    '''

    output = minimize(one_dimensional_function(data_points), x0 = 0., bounds=Bounds(-np.pi, np.pi), method=optimizer)

    return output.fun, output.x

def one_dimensional_function(data_points):
    '''
    Function that performs the molecular GGA 1D local optimization.
    Args:
        data_points (list[float]): list of the measured energies for the GGA method.
    Returns:
        g is the analytic expression of the molecular GGA 1D local optimization
    '''

    new_data_points = []
    for i in range(1,len(data_points)):
        new_data_points.append(data_points[i] - data_points[0])

    M = np.array([[-2, 4, 0, 0], 
                [-1, 1, -1, 1], 
                [-1, 1, 1, -1], 
                [-1/2, 1/4, -np.sqrt(3)/4, np.sqrt(3)/2]])  
    tmp_data_points = np.linalg.inv(M) @ np.array([new_data_points]).T
    g = lambda x: data_points[0] + np.dot(np.array([np.cos(x)-1, (np.cos(x)-1)**2, (np.cos(x)-1)*np.sin(x), np.sin(x)]).T, tmp_data_points)

    return g

#####Ising ADAPT/GGA functions#####
def get_ising_circuits(circuit):
    '''
    Function that creates the several 1D Ising circuits to be measured with the efficient GGA method
    Args:
        circuit (QuantumCircuit): initial quantum circuit.
    Returns:
        list made of 5 different circuits, based on the efficient GGA method for the specific case of 1D transverse-field Ising model
    '''

    circuit_x = circuit.copy()
    circuit_y = circuit.copy()

    for i in range(circuit.num_qubits):
        circuit_x.h(i)
        circuit_y.sdg(i)
        circuit_y.h(i)

    circuit_xz0 = circuit.copy()
    circuit_xz1 = circuit.copy()

    for i in range(circuit.num_qubits):
        if i % 2 == 0:
            circuit_xz0.h(i)
        else:
            circuit_xz1.h(i)
    
    return [circuit, circuit_x, circuit_y, circuit_xz0, circuit_xz1]

def get_ising_counts(counts, qubits):
    '''
    Function that computes the different shot counts needed to evaluate the Ising energy for the efficient GGA method
    Args:
        counts (dict): dictionary containing all the data for the current iteration.
        qubits (int): number of qubits.
    Returns: 
        zis represents the measured shots for the Pauli terms of the form II..IZI..II
        xis represents the measured shots for the Pauli terms of the form II..IXI..II
        zzs represents the measured shots for the Pauli terms of the form II..IZZI..II
        yys represents the measured shots for the Pauli terms of the form II..IYYI..II
        xz0s represents the measured shots for the Pauli terms of the form II..IZXI..II
        xz1s represents the measured shots for the Pauli terms of the form II..IXZI..II
        zxz represents the measured shots for the Pauli terms of the form II..IZXZI.III
    '''

    zis = zis_from_counts(counts[0])
    xis = zis_from_counts(counts[1])

    zzs = zizip1_from_counts(counts[0])
    yys = zizip1_from_counts(counts[2])
    xz0s = zizip1_from_counts(counts[3])
    xz1s = zizip1_from_counts(counts[4])

    zxz = zxz_terms(counts[3], counts[4], qubits)

    return zis, xis, zzs, yys, xz0s, xz1s, zxz

def renormalize_counts(counts):
    '''
    Function that normalize the shot counts with the total number of shots.
    Args:
        counts (dict): dictionary containing all the data for the current iteration.
    Returns:
        counts represents the normalized dictionary of the different shot counts with respective measured states
    '''

    total = sum(list(counts.values()))
    for k in list(counts.keys()):
        counts[k] *= 1./total
    
    return counts

def zis_from_counts(counts):
    '''
    Function that obtains the shot counts for the states corresponding to the Pauli terms of the form II..I\sigma\I..II, where \sigma\ is a non-trivial Pauli matrix.
    Args:
        counts (dict): dictionary containing all the data for the current iteration.
    Returns:
        the wanted results for the terms of the form II..I\sigma\I..II
    '''

    counts = renormalize_counts(counts)
    keys = list(counts.keys())
    n = len(keys[0])

    probas = np.zeros(n)
    for k in keys:
        for qubit in range(n):
            if k[n-1-qubit] == '0':
                probas[qubit] += counts[k]
    
    probas = list(probas)
    results = [2.*p -1 for p in probas]

    return results

def zizip1_from_counts(counts):
    '''
    Function that obtains the shot counts for the states corresponding to the Pauli terms of the form II..I\sigma1\ \sigma2\I..II where both \sigma1\ and \sigma2\ are non-trivial Pauli matrices.
    Args:
        counts (dict): dictionary containing all the data for the current iteration.
    Returns:
        the wanted results for the terms of the form II..I\sigma1\ \sigma2\I..II
    '''

    counts = renormalize_counts(counts)
    keys = list(counts.keys())
    n = len(keys[0])

    probas = np.zeros(n-1)
    for k in keys:
        for qubit in range(n-1):
            if k[n-1-qubit] == k[n-2-qubit]:
                probas[qubit] += counts[k]
    
    probas = list(probas)
    results = [2.*p-1 for p in probas]

    return results

def zxz_term_from_counts(counts):
    '''
    Function that obtains the shot counts for the states corresponding to Pauli terms of the form II..IZXZI..II.
    Args:
        counts (dict): dictionary containing all the data for the current iteration.
    Returns:
        the wanted results for terms of the form II..IZXZI..II
    '''

    counts = renormalize_counts(counts)
    keys = list(counts.keys())
    n = len(keys[0])

    probas= np.zeros(n-2)
    for k in keys:
        for qubit in range(n-2):
            sublist = k[n-3-qubit:n-qubit]
            plus1_measured = sum([int(q) for q in sublist])
            if plus1_measured % 2 == 0:
                probas[qubit] += counts[k]

    probas = list(probas)

    return [2.*p-1 for p in probas]

def zxz_terms(counts1, counts2, n):
    '''
    Function that computes the values for the Pauli terms of the form II..IZXZI..II, from the shot counts of the corresponding states.
    Args:
        counts (dict): dictionary containing all the data for the current iteration.
    Returns:
        the wanted values for Pauli terms of the form II..IZXZI..II
    '''    

    from1 = zxz_term_from_counts(counts1)
    from2 = zxz_term_from_counts(counts2)
    
    vals = []
    for k in range(n-2):
        if k % 2 == 0:
            vals.append(from2[k])
        else:
            vals.append(from1[k])

    return vals

def energy_variance(E, h, J, counts):
    '''
    Function that computes the variance of the energy calculation for the 1D transverse-field Ising model for the GGA-VQE method.
    Args:
        E (float): energy of the Ising system.
        h (float): value of the field constant.
        J (float): value of the coupling constant.
        counts (dict): dictionary contanining all the data for the current iteration.
    Returns:
        the square root of the variance
    '''

    counts = renormalize_counts(counts)
    keys = list(counts.keys())
    n = len(keys[0])

    Variance = 0
    for k in keys:
        local_energy = 0
        for j in range(n):
            if k[j] == '0':
                local_energy += h
            else:
                local_energy -= h
        
        for j in range(n-1):
            if k[j] == k[j+1]:
                local_energy += J 
            else:
                local_energy -= J 
        
        Variance += counts[k]*(local_energy-E)**2
    
    return np.sqrt(Variance)

def hand_optimization(A, B):
    '''
    1D optimization for the 1D transverse-field Ising model used in the GGA-VQE method.
    Args:
        A (float): energy value of the first part of the Hamiltonian.
        B (float): energy value of the second part of the Hamiltonian.
    Returns:
        the computed 1D optimization function.
    '''

    a, b = A, -.5*B

    if B == 0.:
        thetaopt = -np.pi/2
    else:
        thetaopt = np.arctan(a/b)
    
    Leffopt = a*np.sin(thetaopt) + b*np.cos(thetaopt)

    if Leffopt >= 0:
        Leffopt *= 1
        thetaopt += np.pi

    return Leffopt + .5*B, thetaopt

######Data retrieving functions######
def get_complete_eval_counts(count):
    '''
    Function that computes the number of cost function evaluation in a (GGA)Adapt-VQE simulation
    Args:
        counts (dict): dictionary containing all the data measured for each iteration.
    Returns:
        eval counts(list): list containing one element, being the number of cost function evaluation. 
    '''

    eval_counts = []
    for i in range(len(count)):
        if count[i] == 1:
            if i == 0:
                pass 
            else:
                eval_counts.append(tmp_eval_counts)
            tmp_eval_counts = 0
        tmp_eval_counts +=1

    eval_counts.append(tmp_eval_counts)

    return eval_counts

def get_real_mean_values(count, value):
    '''
    Creates a list containing the energy value for each iteration of an (GGA)Adapt-VQE simulation.
    Args:  
        count (dict): dictionary containing all data for each iteration.
        value (list): list of the raw measured energies.
    Returns:
        real_values (list[float]): list where each element is the energy value for a specific iteration, sorted from the first to the last iteration.
    '''

    real_values = []
    for item in count:
        if item == count[0]:
            real_values.append(value[item-1])
        else:
            real_values.append(value[prev_item + item -1])
        prev_item = item

    return real_values

def get_real_st_dev(count, std_dev):
    '''
    Creates a list containing the standard deviation of the energy value for each iteration of an Adapt-VQE simulation.
    Args:
        count (dict): dictionary containing all the data for each iteration.
        std_dev (list): list of the raw standard deviations.
    Returns:
        real_std (list[float]): list where each element is the standard deviation of the energy value for a specific iteration, sorted from the first to the last iteration.
    '''

    real_std = []
    for item in count:
        if item == count[0]:
            real_std.append(std_dev[item -1])
        else:
            real_std.append(std_dev[prev_item + item - 1])
        prev_item = item
    
    return real_std

def get_gga_mean_variances(metadata):
    '''
    Creates a list containing the mean variance value of the energy evaluation of each iteration in an GGA-Adapt-VQE simulation.
    By mean variance it is meant the mean value of the variance of each of the five points used in the GGA method.
    Args:
        metadata (dict): dictionary containing all the metadata for each iteration.
    Returns:
        variances (list[float]): list where each element is the mean variance of the energy evaluation for a specific iteration, sorted from the first to the last iteration.
    '''

    variances = []
    for i in range(len(metadata)):
        tmp_variance = 0
        for j in range(len(metadata[i])):
            tmp_variance += metadata[i][j]['variance']
        variances.append(tmp_variance/5)
    
    return variances

def get_gga_variances(metadata):
    '''
    Creates a list containing the extrapolated variance value of the energy evaluation for each iteration of a GGA-Adapt-VQE simulation.
    By extrapolated variance it is meant the variance corresponding to the optimal parameter value, extrapolated from the variances of the 5 optimal parameter values used in the GGA method.
    Args:
        metadata (dict): dictionary containing all the metadata for each iteration.
    Returns:
        variances (list[float]): list where each element is the extrapolated variance of the energy evaluation at a specific iteration, sorted from the first to the last iteration.
    '''

    variances = []
    for i in range(len(metadata)):
        tmp_variance = []
        for j in range(len(metadata[i])):
            tmp_variance.append(metadata[i][j]['variance'])
        variances.append(tmp_variance)
    
    return variances

#####Restart functions ##### 
def create_restart_file(path, indices, optangles):
    '''
    Creates a file to be used as a restart for a new simulation.
    Args:
        path (str): path to the where the restart file should be stored.
        indices (list[int]): ordered list of the indices of each appended operator (with respect to their order in the Pool)
        optangles (list[float]): ordered list of the optimal parameter values of each appended operator (with respect to their order in the Pool)
    '''

    restart_file = open(path + '/restart.txt', "w")
    restart_file.writelines(str(indices))
    restart_file.writelines('\n')
    restart_file.writelines(str(optangles))
    restart_file.close()

def get_restart_state(path, qubits):
    '''
    Fetch the restart data of a previous (GGA)Adapt-VQE simulation, namely the indices of each appended operators and theirs associated optimal parameter values.
    To be used to the creation of an initial state restarting this previous simulation.
    Args:
        path (str): string representation of the path to the restart file.
        qubits (int): number of qubits.
    Returns:
        optangles (list[float]): list containing the optimal parameter values associated to each appended operator.
        indices (list[float]): list containing the indices of each appended operators.
    '''

    print("RESTART!!!!")

    angles = []
    indices = []

    tmp_list = []
    with open(path, 'r') as file:
        for line in file:
            tmp_list.append(line)

    indices = ast.literal_eval(tmp_list[0])
    optangles = ast.literal_eval(tmp_list[1])
        
    return optangles, indices

#####Multiprocessing functionalities#####
def chunk_exci_pool(excitation_pool, num_jobs):
    '''
    Creates a chunck of the excitation pool for parallelizing the gradient calculations in an (GGA)Adapt-VQE simulation.
    Depends on the number of CPUs given for the computation.
    Args:
        excitation_pool (list[BaseOperator]): list of all the excitations in the operator pool.
        num_jobs (int): number of CPUs affected.
    Returns:
        the chunk list
    '''

    deque_obj = deque(excitation_pool)
    chunk_size = int(len(excitation_pool)/num_jobs) +1
    while deque_obj:
        chunk = []
        for _ in range(chunk_size):
            if deque_obj:
                chunk.append(deque_obj.popleft())           
        yield chunk 

#####Fermionic pool specificities (WIP)#####
def create_single_pauli_operator(nqubits, excitation_tuple, excitation_string):
    '''
    Creates a Pauli string of a single fermionic excitation operator (UCCSD framework) in the Qiskit language.
    Args:
        nqubits (int): number of qubits.
        excitation_tuple (tuple): tuple of the qubits non-trivially involved in the single pauli operator with their corresponding single-qubit pauli operator.
        excitation_string (str): string representation of the Pauli string.
    Returns:
        A Qiskit PauliOp of this single fermionic excitation operator's Pauli string.
    '''

    Xs, Ys, Zs = create_pauli_operators(nqubits)

    bottom_qubit = excitation_tuple[0][0]
    top_qubit = excitation_tuple[1][0]

    if excitation_string[0] == 'X':
        pauli_string = Xs[bottom_qubit] 
    else:
        pauli_string = Ys[bottom_qubit]
    for i in range(bottom_qubit+1, top_qubit):
        pauli_string = pauli_string & Zs[i]
    if excitation_string[1] == 'Y':
        pauli_string = pauli_string & Ys[top_qubit]
    else:
        pauli_string = pauli_string & Xs[top_qubit]

    return PauliOp(pauli_string, 1.0)

def create_double_pauli_operator(nqubits, excitation_tuple, excitation_string):
    '''
    Creates a Pauli string of a double fermionic excitation operator (UCCSD framework), in the Qiskit language.
    Args:
        nqubits (int): number of qubits.
        excitation_tuple (tuple): tuple of the qubits non-trivially involved in the double pauli operator with their corresponding single-qubit pauli operator.
        excitation_string (str): string representation of the Pauli string.
    Returns:
        A Qiskit PauliOp of this double fermionic excitation operator's Pauli string.
    '''

    Xs, Ys, Zs = create_pauli_operators(nqubits)

    qubit1 = excitation_tuple[0][0]
    qubit2 = excitation_tuple[0][1]
    qubit3 = excitation_tuple[1][0]
    qubit4 = excitation_tuple[1][1]

    if excitation_string[0] == 'X':
        pauli_string = Xs[qubit1]
    else:
        pauli_string = Ys[qubit1]

    for i in range(qubit1 + 1, qubit2):
        pauli_string = pauli_string & Zs[i]
    
    if excitation_string[1] == 'X':
        pauli_string = pauli_string & Xs[qubit2]
    else:
        pauli_string = pauli_string & Ys[qubit2]
    
    for i in range(qubit2 + 1, qubit3):
        pauli_string = pauli_string & Zs[i]
    
    if excitation_string == 'X':
        pauli_string = pauli_string & Xs[qubit3]
    else:
        pauli_string = pauli_string & Ys[qubit3]
    
    for i in range(qubit3 + 1, qubit4):
        pauli_string = pauli_string & Zs[i]
    
    if excitation_string == 'X':
        pauli_string = pauli_string & Xs[qubit4]
    else:
        pauli_string = pauli_string & Ys[qubit4]
    
    return PauliOp(pauli_string, 0.125)

#####Plot functions#####
def plot_comparison_cost(y_adapt, y_gga, y_adapt_noiseless, y_gga_noiseless, v_adapt, v_gga, fci, down_lim, upper_lim, name, method):
    '''
    Function to plot a comparison between Adapt-VQE and GGA-Adapt-VQE simulations for a molecular system in terms of Energy values for each iteration.        
    Args:
        y_adapt (list[float]): list of the values obtained from a noisy/real Adapt-VQE simulation.
        y_gga (list[float]): list of the values obtained from a noisy/real GGA-VQE simulation.
        y_adapt_noiseless (list[float]): list of the values obtained from a noiseless Adapt-VQE simulation.
        y_gga_noiseless (list[float]): list of the values obtained from a noiseless GGA-VQE simulation.
        v_adapt (list[float]): list of the values of the variance for each Adapt-VQE point.
        v_gga (list[float]): list of the values of the variance for each GGA-VQE point.
        fci (float): numerical value of the FCI energy of the molecule.
        down_lim (float): numerical value of the minimal energy for centering the plot.
        upper_lim (float): numerical value of the maximal energy for centering the plot.
        name (str): Name of the molecule.
        method (str): Name of the method used in the simulation.
    '''

    #Number of the abscisse points
    xmax = min(len(y_adapt), len(y_gga), 30)

    mpl.rc('font', family='serif', size=10)
    # Set the size of the plot
    plt.figure(figsize=(7, 6))

    # Plot the line
    plt.plot(np.arange(len(y_adapt)), y_adapt, label='ADAPT', linewidth=2, color='black')
    plt.plot(np.arange(len(y_gga)), y_gga, label='GGA', linewidth=2, color='green')
    plt.plot(np.arange(len(y_adapt_noiseless)), y_adapt_noiseless, label='ADAPT corrected', linewidth=2, color='blue')
    plt.plot(np.arange(len(y_gga_noiseless)), y_gga_noiseless, label='GGA corrected', linewidth=2, color='red')
    plt.plot(np.arange(xmax), [fci + 0.001 for i in range(xmax)], linewidth=2, linestyle='--')
    # Add error bars
    # plt.errorbar(np.arange(len(y_adapt)), y_adapt, yerr=v_adapt, fmt='o', color='black', ecolor='red', capsize=5, label='Standard Deviation')
    # plt.errorbar(np.arange(len(y_gga)), y_gga, yerr=v_gga, fmt='o', color='green', ecolor='red', capsize=5, label='Standard Deviation')
    
    plt.xlim(0, xmax)
    plt.ylim(round(down_lim, 2), round(upper_lim, 2))
    plt.xlabel('Iterations', fontsize=18)
    plt.ylabel('Energy', fontsize=18)
    
    # Create the legend
    plt.legend(loc=(0.07, 0.75), edgecolor='black', title=name)
    plt.grid(True, color='black', linestyle=':', linewidth=0.5)

    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    
    # Save the figure
    if method == 'vqe':
        complete_name = name + ' (noisy optimization)'
        plt.title(complete_name)
    else:
        complete_name = name + ' (noisy optimization + gradient screening)'
        plt.title(complete_name)
    plt.savefig(complete_name + '.eps')

def molecular_specific_constants(molecule_name):
    '''
    Provides mandatory constants for the data analysis (plots) of molecular systems, simulated through (GGA)Adapt-VQE simulation.
    Args:
        molecule_name (str): name of the molecule.
    Returns:
        fci, the energy of the FCI state (fetched from PySCF).
        hf, the energy of the HF state (fetched from PySCF).
        down_lim, the minimal value for the plot.
        upper_lim, the maximal value for the plot.
    '''

    if molecule_name == 'LIH':
        fci = -7.8237238834677
        hf = -7.770873669221907
        down_lim = -7.83
        upper_lim = -7.76
    elif molecule_name == 'H6':
        fci = -2.8009589
        hf = -1.9706022459979957
        down_lim = -2.82
        upper_lim = -1.90
    elif molecule_name == 'H2O':
        fci = -74.28484358473612
        hf = -74.2658431001177
        down_lim = -74.30
        upper_lim = -74.26
    elif molecule_name == 'H2':
        fci = -1.01431027471335
        hf = -0.9162712476952433
        down_lim = -1.02
        upper_lim = -0.90
    elif molecule_name == 'He':
        fci = -2.870162138900823
        hf = -2.8551604261544465
        down_lim = -2.88
        upper_lim = -2.85
    else:
        raise ValueError
    
    return fci, hf, down_lim, upper_lim