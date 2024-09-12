"""An implementation of the AdaptVQE algorithm."""
from __future__ import annotations

from collections.abc import Sequence
from enum import Enum

import os
import re
import logging
from typing import Any
import copy

import multiprocessing as mp

import numpy as np

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.primitives import Estimator
from qiskit.quantum_info import commutator as qiskit_com
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.circuit.library import EvolvedOperatorAnsatz
from qiskit_algorithms.utils.validation import validate_min
from qiskit_algorithms.exceptions import AlgorithmError
from qiskit_algorithms.minimum_eigensolvers import MinimumEigensolver
from qiskit_algorithms.variational_algorithm import VariationalAlgorithm

from vqe import VQE, VQEResult
from observables_evaluator import estimate_observables
import UsefulFunctions as uf

logger = logging.getLogger(__name__)

class TerminationCriterion(Enum):
    """A class enumerating the various finishing criteria."""

    CONVERGED = "Threshold converged"
    MAXIMUM = "Maximum number of iterations reached"

class AdaptVQE(VariationalAlgorithm, MinimumEigensolver):
    """
    The following attributes can be set via the initializer but can also be read and updated once
    the AdaptVQE object has been constructed.

    Attributes:
        system_config: a class containing all informations concerning the studied system, which can be either 
            molecular or an Ising 1D chain. All of its attributes is passed to the AdaptVQE class.
        adapt_config: a class containing all informations of the simulation such as the optimizer, the operator pool,
            thresholds and so on. All of its attributes is passed to the AdaptVQE class.
    """

    def __init__(self, system_config, adapt_config) -> None:

#<<<<<<<<<<<<<<<<<<< Attributes from setup classes >>>>>>>>>>>>>>>>>>>#

        system_attributes = [attr for attr in dir(system_config) if '__' not in attr]
        for attr in system_attributes:
            setattr(self, attr, system_config.__dict__[attr])
        
        adapt_attributes = [attr for attr in dir(adapt_config) if '__' not in attr]
        for attr in adapt_attributes:
            setattr(self, attr, adapt_config.__dict__[attr])

#<<<<<<<<<<<<<<<<<<< VQE Solver definition >>>>>>>>>>>>>>>>>>>#

        #Qiskit ansatz
        ansatz = EvolvedOperatorAnsatz(operators=self.generators, initial_state=self.initial_state)

        #Callback function
        def store_intermediate_results(eval_count, parameters, mean, std):

            counts.append(eval_count)
            values.append(mean)
            std_dev.append(std)
        
        counts = []
        values = []
        std_dev = []

        self.adapt_counts = counts
        self.adapt_values = values
        self.adapt_std_dev = std_dev

        #General set-up of the VQE solver
        self.estimator = Estimator()
        self.solver = VQE(self.estimator,
            ansatz=ansatz,
            optimizer=self.optimizer,
            shots=self.shots,
            callback=store_intermediate_results
            )

#<<<<<<<<<<<<<<<<<<< ADAPT general attributes >>>>>>>>>>>>>>>>>>>#

        validate_min("gradient_threshold", self.gradient_threshold, 1e-15)
        validate_min("eigenvalue_threshold", self.eigenvalue_threshold, 1e-15)

        self._tmp_ansatz: EvolvedOperatorAnsatz | None = None
        self.tmp_ansatz: QuantumCircuit | None = None
        self._excitation_list: list[SparsePauliOp] = []
        self._excitation_pool: list[BaseOperator] = []

        self.noiseless_energies: list[float] = []
        self.variational_energies: list[float] = []

        if self.adapt1d is True:
            self.dict_parameters = {}
        elif self.adapt1d is False:
            self.dict_parameters = None

#<<<<<<<<<<<<<<<<<<< GGA specific attributes >>>>>>>>>>>>>>>>>>>#

        if self.system == 'Molecule':
            self.num_se = len(self.pool.single_generators)
        self.spaced_thetas = [0, np.pi, np.pi/2, -np.pi/2, np.pi/3]
        self.energy_variances: list[float] = []
        self.metadata: lisŧ[dict] = []
        self.min_metadata: lisŧ[dict] = []
        self.energy_drops: list[float] = []
        self.stored_angles: list[float] = []

        #1D GGA Reoptimisation
        self.reoptimised_minimums: list[float] = []
        self.reoptimised_thetas: list[float] = []

#<<<<<<<<<<<<<<<<<<< Restart specific attributes (WIP) >>>>>>>>>>>>>>>>>>>#
        if self.shots is None:
            self.restart_path = os.path.join(os.path.dirname(__file__), f'simulations/data/{self.name}/noiseless/restart.txt')
        else:
            self.restart_path = os.path.join(os.path.dirname(__file__), f'simulations/data/{self.name}/{self.shots}/restart.txt')

        self.indices: list[int] = []
        self.optangles: list[float] = []

    
    @property
    def initial_point(self) -> Sequence[float] | None:
        """Returns the initial point of the internal :class:`~.VQE` solver."""
        return self.solver.initial_point

    @initial_point.setter
    def initial_point(self, value: Sequence[float] | None) -> None:
        """Sets the initial point of the internal :class:`~.VQE` solver."""
        self.solver.initial_point = value


    def _compute_gradients_adapt(self, theta: list[float], operator: BaseOperator) -> list[tuple[complex, dict[str, Any]]]:
        """
        Args:
            theta (list[float]): List of (up to now) optimal parameters.
            operator (BaseOperator): Operator whose gradient needs to be computed.
        Returns:
            List of pairs consisting of the computed gradient and excitation operator.
        """
        ## The excitations operators are applied later as exp(i*theta*excitation). For this commutator, we need to explicitly pull in the imaginary phase.
        
        if self.cpu_count == 1:
            commutators = [1j * qiskit_com(operator, exc) for exc in self.generators]

            #Distinction between native Adapt-VQE simulation and 1D Adapt-VQE one
            if self.dict_parameters is None or theta == []:
                if self.noisy == 'vqe':
                        res = estimate_observables(self.solver.estimator, self.solver.ansatz, commutators, theta)
                elif self.noisy == 'full':  
                    res = estimate_observables(self.solver.estimator, self.solver.ansatz, commutators, theta, shots=self.shots)
                else:
                    raise ValueError
            else:
                if self.noisy == 'vqe':
                        res = estimate_observables(self.solver.estimator, self.solver.ansatz, commutators, [theta[-1]])
                elif self.noisy == 'full':  
                    res = estimate_observables(self.solver.estimator, self.solver.ansatz, commutators, [theta[-1]], shots=self.shots)
                else:
                    raise ValueError
        elif self.cpu_count > 1:
                
            global compute_gradients_parallel
            def compute_gradients_parallel(index, operator, generators, theta):
                res = []
                commutators = [1j * qiskit_com(operator, exc) for exc in generators]
                for commutator in commutators:
                    if self.noisy == 'vqe':
                        results = estimate_observables(self.solver.estimator, self.solver.ansatz, [commutator], theta)
                    elif self.noisy =='full':
                        results = estimate_observables(self.solver.estimator, self.solver.ansatz, [commutator], theta, shots=self.shots)
                    res.append(results[0])
                    
                return (index, res)  

            exc_pool = list(uf.chunk_exci_pool(self.generators, self.cpu_count))

            if self.verbose:
                for i in range(len(exc_pool)):
                    print(f"length of the chuncked pool for the cpu {i}: ", len(exc_pool[i]))
            
            #Parallel computation of Adapt-VQE gradients, only for the native version (i.e. full optimization)
            with mp.Pool(processes=len(exc_pool)) as pool:
                pool_results = pool.starmap(compute_gradients_parallel, [(i, operator, exc_pool[i], theta) for i in range(len(exc_pool))])
            res = []
            for i in range(len(pool_results)):
                res += pool_results[i][1]

        return res

    def _compute_gradients_gga(self, theta: list[float], operator: BaseOperator):
        """
        Function used to select the CPU method of computing GGA gradients-like values.
        Args:
            theta (list[float]): List of (up to now) optimal parameters.
            operator (BaseOperator): operator whose gradient needs to be computed.
        """

        if self.cpu_count == 1:
            return self._compute_gradients_gga_serial(theta, operator)
        elif self.cpu_count > 1:
            return self._compute_gradients_gga_parallel(theta, operator) 
        else:
            raise ValueError
    
    def _compute_gradients_gga_serial(self, theta: list[float], operator: BaseOperator) -> list[tuple[complex, dict[str, Any]]]:
        """
        Function computing gradients-like values in the GGA-VQE method using 1CPU, for molecular sytems and the 1D transverse-field Ising model.
        For molecular systems only, the two following options are available:
            method based on explicit circuits construction, done previously in the Pool construction (see OperatorPool)
            method based on Qiskit evolution framework, using LieTrotter decomposition and associated evolution (native EvolvedOperatorAnsatz method)
        Args:
            theta (list[float]): List of (up to now) optimal parameters.
            operator (BaseOperator): operator whose gradient needs to be computed.
        Returns:
            List of pairs consisting of the computed gradient and excitation operator.
        """
        
        theta.append(0.0)
        if self.system == 'Molecule':
            res = []
            for generator in range(len(self.generators)):
                data_points = []
                metadata = []

                if self.method == 'circuit':
                    tmp_ansatz = self.solver.ansatz.copy()
                    tmp_ansatz.append(self.excitations[generator], tmp_ansatz.qubits)
                    for angle in self.spaced_thetas:
                        theta[-1] = angle
                        results = estimate_observables(self.solver.estimator, tmp_ansatz, [operator], theta, shots=self.shots)
                        data_points.append(results[0][0])
                        metadata.append(results[0][1])
                elif self.method == 'evolution':
                    tmp_excitation_list = self._excitation_list.copy()
                    tmp_excitation_list.append(self.generators[generator])
                    self._tmp_ansatz.operators = tmp_excitation_list
                    for angle in self.spaced_thetas:
                        theta[-1] = angle
                        results = estimate_observables(self.solver.estimator, self._tmp_ansatz, [operator], theta, shots=self.shots)
                        data_points.append(results[0][0]) 
                        metadata.append(results[0][1])
                
                expected_minimum, theta_min = uf.compute_minimum_in_direction(data_points, self.optimizer)
                res.append((expected_minimum, self.generators[generator], theta_min))
                self.metadata.append(metadata)
            
            return res

        elif self.system == 'Ising':
            #QPU results
            tmp_ansatz = self.solver.ansatz.copy()
            counts = uf.measure_circuits(uf.get_ising_circuits(tmp_ansatz), theta, self.shots)
            zis, xis, zzs, yys, xz0s, xz1s, zxz = uf.get_ising_counts(counts, self.nqubits)

            #Energy calculation
            energy = self.h * sum(xis) + self.J * sum(zzs)

            #Variance calculation
            xvar = uf.energy_variance(self.h * sum(xis), np.abs(self.h), 0., counts[1])
            yvar = uf.energy_variance(self.J * sum(zzs), 0., np.abs(self.J), counts[0])
            if self.shots is not None:
                xvar = xvar/np.sqrt(self.shots -1)
                yvar = yvar/np.sqrt(self.shots -1)

            energy_drops = []
            stored_angles = []
            #1D optimization 
            for j in range(self.nqubits):
                if j == self.nqubits - 1:
                    A = self.h * zis[j] - self.J * xz1s[j-1]
                    B = -2 * self.h * xis[j] - 2 * self.J * zzs[j-1]
                else:
                    A = self.h * zis[j]
                    B = -2* (self.h * xis[j] + self.J * zzs[j])

                    if j % 2 == 0:
                        A -= self.J * xz0s[j]
                    else:
                        A -= self.J * xz1s[j]
                    
                    if j > 0:
                        B -= 2. * self.J * zzs[j-1]
                        if j % 2 == 0:
                            A -= self.J * xz0s[j-1]
                        else:
                            A -= self.J * xz1s[j-1]
                
                opt_eigenvalue, opt_angle = uf.hand_optimization(A, B)
                energy_drops.append(opt_eigenvalue)
                stored_angles.append(opt_angle)

            for j in range(self.nqubits, 2*self.nqubits - 1):
                k = j % (self.nqubits)

                A = self.h * (zzs[k] - yys[k]) - self.J * xis[k+1]
                B = -2 * self.h * (xis[k] + xis[k+1]) - 2 * self.J * zzs[k]

                if k < self.nqubits -2:
                    A -= self.J * zxz[k]
                    B -= 2 * self.J * zzs[k+1]
                
                opt_eigenvalue, opt_angle = uf.hand_optimization(A, B)
                energy_drops.append(opt_eigenvalue)
                stored_angles.append(opt_angle)
            
            min_index = energy_drops.index(min(energy_drops))
            opt_theta = stored_angles[min_index]
            
            return energy, min_index, opt_theta

    def _compute_gradients_gga_parallel(self, theta: list[float], operator: BaseOperator) -> list[tuple[complex, dict[str, Any]]]:
        """
        Function computing gradients-like values in the GGA-VQE method using multiple CPUs, for molecular sytems.
        The two following options are available:
            method based on explicit circuits construction, done previously in the Pool construction (see OperatorPool)
            method based on Qiskit evolution framework, using LieTrotter decomposition and associated evolution (native EvolvedOperatorAnsatz method)
        Args:
            theta: List of (up to now) optimal parameters.
            operator: operator whose gradient needs to be computed.
        Returns:
            List of pairs consisting of the computed gradient and excitation operator.
        """

        theta.append(0.0)
        if self.system == 'Molecule':
            res = []
            metadatas = []

            global gradients_gga_parallel_evol
            def gradients_gga_parallel_evol(index, operator, generators, theta):
                
                for generator in generators:
                    data_points = []
                    metadata = []
                    self._tmp_ansatz.operators = generator
                    for angle in self.spaced_thetas:
                        theta[-1] = angle
                        results = estimate_observables(self.solver.estimator, self._tmp_ansatz, [operator], theta, shots=self.shots)
                        data_points.append(results[0][0])
                        metadata.append(results[0][1])
                    
                    expected_minimum, theta_min = uf.compute_minimum_in_direction(data_points, self.optimizer)
                    res.append((expected_minimum, generator, theta_min))
                    metadatas.append(metadata)

                return (index, res, metadatas)

            global gradients_gga_parallel_circ
            def gradients_gga_parallel_circ(index, operator, circuits, generators, theta):
                data_points = []
                metadata = []
                for circuit in range(len(circuits)):
                    for angle in self.spaced_thetas:
                        theta[-1] = angle
                        results = estimate_observables(self.solver.estimator, circuit, [operator], theta, shots=self.shots)
                        data_points.append(results[0][0])
                        metadata.append(results[0][1])
                    
                    expected_minimum, theta_min = uf.compute_minimum_in_direction(data_points, self.optimizer)
                    res.append((expected_minimum, generators[circuit], theta_min))
                    metadatas.append(metadata)

                return (index, res, metadatas)

            tmp_ansatzes = [] 
            if self.method =='evolution':
                for generator in range(len(self.generators)):
                    tmp_excitation_list = self._excitation_list.copy()
                    tmp_excitation_list.append(self.generators[generator])
                    tmp_ansatzes.append(tmp_excitation_list)
                exc_pool = list(uf.chunk_exci_pool(tmp_ansatzes, self.cpu_count))
                with mp.Pool(processes=len(exc_pool)) as pool:
                    pool_results = pool.starmap(gradients_gga_parallel_evol, [(i, operator, exc_pool[i], theta) for i in range(len(exc_pool))])
                
                for i in range(len(pool_results)):
                    for j in range(len(pool_results[i][1])):
                        res.append(pool_results[i][1][j])
                        self.metadata.append(pool_results[i][2][j])

            elif self.method == 'circuit':
                tmp_ansatzes.append(self.solver.ansatz.copy().append(self.excitations[generator], self.solver.ansatz.qubits) for generator in range(len(self.generators)))
                exc_pool = list(uf.chunk_exci_pool(tmp_ansatzes), self.cpu_count)
                with mp.Pool(processes=len(exc_pool)) as pool:
                    pool_results = pool.starmap(gradients_gga_parallel_circ, [(i, operator, exc_pool[i], theta) for i in range(len(exc_pool))])
                
                for i in range(pool_results):
                    for j in range(len(i[1])):
                        res.append(i[1][j])
                        self.metadata.append(i[2][j])

            return res

        elif self.system == 'Ising':
            print('No implementation of a parallel computation for a GGA-VQE simulation for an Ising model is yet implemented!!!!!\n')
            raise ValueError('!!END OF SIMULATION!!')

    def compute_minimum_eigenvalue(self, operator: BaseOperator) -> AdaptVQEResult:
        """
        Args:
            operator (BaseOperator): Operator whose minimum eigenvalue we want to find.

        Raises:
            TypeError: If an ansatz other than :class:`~.EvolvedOperatorAnsatz` is provided.
            AlgorithmError: If all evaluated gradients lie below the convergence threshold in
            the first iteration of the algorithm.

        Returns:
            self.indices (list[int]): A list of the indices of the appended operators (with respect to their position in the Pool)
            self.optangles (list[float]): A list of the optimal parameter values for each appended operators
            result: A AdaptVQEResult, being a modification of the VQEResult to also include runtime information about the AdaptVQE algorithm like the number of iterations, termination criterion, and the final maximum gradient.
            self.noiseless_energies (list[float]): A list containing the energy at each iteration, measured in an exact setup (i.e. without any noise)
            self.reoptimised_minmums (list[float]): A list containing the energy at each iteration, with a 1D reoptimisation process (for the GGA algorithm)
            self.reoptimised_thetas (list[float]): A list containing the optimal parameter values at each iteration, with a 1D reoptimisation process (for the GGA algorithm)
        """
        if not isinstance(self.solver.ansatz, EvolvedOperatorAnsatz):
            raise TypeError("The AdaptVQE ansatz must be of the EvolvedOperatorAnsatz type.")

        # Overwrite the solver's ansatz with the initial state
        self._tmp_ansatz = self.solver.ansatz
        self.solver.ansatz = self._tmp_ansatz.initial_state
        self._excitation_pool = self._tmp_ansatz.operators

        prev_raw_vqe_result: VQEResult | None = None
        raw_vqe_result: VQEResult | None = None
        theta: list[float] = []
        max_grad: tuple[complex, dict[str, Any] | None] = (0.0, None)
        history: list[complex] = []

        if self.restart is True:
            if self.method == 'circuit':
                self.optangles, self.indices = uf.get_restart_state(self.restart_path, self.nqubits)
                for i in self.indices:
                    self.solver.ansatz.append(self.excitations[i].assign_parameters([Parameter('c_'+str(i))]), range(self.nqubits))
            elif self.method == 'evolution':
                self.optangles, self.indices = uf.get_restart_state(self.restart_path, self.nqubits)
                for i in self.indices:
                    self._excitation_list.append(self.generators[i])
                self._tmp_ansatz.operators= self._excitation_list
                self.solver.ansatz = self._tmp_ansatz
            if self.adapt1d is True:
                for i in range(len(self.indices)):
                    self.dict_parameters.update({self.solver.ansatz.parameters[i]: self.optangles[i]})
                self.solver.ansatz.assign_parameters(self.dict_parameters, inplace=True)

        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1
            logger.info("--- Iteration #%s ---", str(iteration))

            if self.algorithm == 'gga':

                #Computing gradients
                logger.debug("Computing gradients")
                if self.system == 'Molecule':
                    gradients = self._compute_gradients_gga(theta, operator)
                    min_index, minimum = max(enumerate(gradients), key=lambda item: np.abs(item[1][0]))
                    theta_min = gradients[min_index][-1][0]
                    self.min_metadata.append(self.metadata[min_index])
                    
                    if self.verbose:
                        print("minimum:", minimum[0])
                        print("metadata for best gradient:", self.min_metadata[-1])
                elif self.system == 'Ising':
                    minimum, min_index, theta_min = self._compute_gradients_gga(theta, operator)
                    
                    if self.verbose:
                        print("minimum:", minimum)

                self.indices.append(min_index)

                #Inlive log prints
                if self.verbose:
                    print("minimal index ", min_index)
                    print("theta_min:", theta_min)
                    print("New operator added to the ansatz: ", self.generators[min_index])

                #Add new excitation to the ansatz
                logger.info("Adding new operator to the ansatz: %s", str(self.generators[min_index]))
                if self.method == 'circuit':
                    self._excitation_list.append(self.excitations[min_index].copy('c_'+str(iteration)))
                    self.solver.ansatz.append(self.excitations[min_index].assign_parameters([Parameter('c_'+str(iteration))]), range(self.nqubits))
                elif self.method == 'evolution':
                    self._excitation_list.append(self.generators[min_index])
                    self._tmp_ansatz.operators= self._excitation_list
                    self.solver.ansatz = self._tmp_ansatz
                else:
                    raise ValueError 

                #Updating the value of the angles' list
                theta[-1] = theta_min
                self.solver.initial_point = theta

                #Backward optimization
                if self.reoptimisation and iteration > 1:
                    new_minimum, new_theta = self.backward_optimization(theta.copy(), operator)
                    minimum = list(minimum)
                    minimum[0] = new_minimum
                    minimum = tuple(minimum)
                    theta = new_theta

                #Calculating the "corrected" eigenvalue
                self.noiseless_energies.append(self.compute_noiseless_energy(theta, operator))
                if self.verbose:
                    print("noiseless energy", self.noiseless_energies[-1])

                history.append(minimum)
                logger.info("Current eigenvalue: %s", str(minimum))
                print(f"Iteration {iteration} done")

                #Checking convergence
                if iteration > 1:
                    if self.system == 'Molecule':
                        eigenvalue_diff = np.abs(history[-2][0] - history[-1][0])
                    elif self.system == 'Ising':
                        eigenvalue_diff = np.abs(history[-2] - history[-1])
                    if eigenvalue_diff < self.eigenvalue_threshold:
                        logger.info("GGA AdaptVQE terminated successfully with a final change in eigenvalue: %s", str(eigenvalue_diff))
                        termination_criterion = TerminationCriterion.CONVERGED
                        # logger.debug("Reverting the addition of the last excitation to the ansatz since it resulted in a change of the eigenvalue below the configured threshold.")
                        break 

            elif self.algorithm == 'adapt':
                #Compute gradients
                logger.debug("Computing gradients")
                gradients = self._compute_gradients_adapt(theta, operator)
                max_grad_index, max_grad = max(enumerate(gradients), key=lambda item: np.abs(item[1][0]))

                logger.info("Found maximum gradient %s at index %s", str(np.abs(max_grad[0])), str(max_grad_index))
                if self.verbose:
                    print("Index of the appended operator: ", max_grad_index)
                self.indices.append(max_grad_index)

                #Checking gradient threshold
                if np.abs(max_grad[0]) < self.gradient_threshold:
                    if iteration == 1:
                        raise AlgorithmError(
                            "All gradients have been evaluated to lie below the convergence threshold during the first iteration of the algorithm."
                            "Try to either tighten the convergence threshold or pick a different ansatz."
                        )
                    logger.info("AdaptVQE terminated successfully with a final maximum gradient: %s", str(np.abs(max_grad[0])))
                    termination_criterion = TerminationCriterion.CONVERGED
                    break

                #Add new excitation to the ansatz
                logger.info("Adding new operator to the ansatz: %s", str(self.generators[max_grad_index]))
                theta.append(0.0)

                #self._excitation_list.append(self.generators[max_grad_index])
                if self.method == 'circuit':
                    self._excitation_list.append(self.excitations[max_grad_index].copy('c_'+str(iteration)))
                    self.solver.ansatz.append(self._excitation_list[-1].assign_parameters([Parameter('c_'+str(iteration))]), self.solver.ansatz.qubits)
                elif self.method == 'evolution':
                    self._excitation_list.append(self.generators[max_grad_index])
                    self._tmp_ansatz.operators = self._excitation_list
                    self.solver.ansatz = self._tmp_ansatz
                else:
                    raise ValueError
                                
                #Set up and performing the VQE iteration
                self.solver.dict_initial_point = self.dict_parameters
                self.solver.initial_point = theta
                prev_raw_vqe_result = raw_vqe_result
                raw_vqe_result = self.solver.compute_minimum_eigenvalue(operator)

                if self.dict_parameters is None:
                    theta = raw_vqe_result.optimal_point.tolist()
                else:
                    if iteration == 1:
                        theta = raw_vqe_result.optimal_point.tolist()
                    else:
                        theta[-1]= raw_vqe_result.optimal_point.tolist()[0]
                    self.dict_parameters.update({self.solver.ansatz.parameters[-1]: theta[-1]})

                #Calculating "corrected" eigenvalues
                if self.dict_parameters is None:
                    self.noiseless_energies.append(self.compute_noiseless_energy(theta, operator))
                else:
                    self.noiseless_energies.append(self.compute_noiseless_energy(theta[-1], operator))

                #Inlive log prints
                if self.verbose:
                    print("optimal current theta: ", theta)
                    print("minimum eigenvalue:", raw_vqe_result.eigenvalue)
                    print("current cost function eval: ", raw_vqe_result.cost_function_evals)
                    print("noiseless energy", self.noiseless_energies[-1])

                #Checking convergence based on the change in eigenvalue
                if iteration > 1:
                    eigenvalue_diff = np.abs(raw_vqe_result.eigenvalue - history[-1])
                    if eigenvalue_diff < self.eigenvalue_threshold:
                        logger.info("AdaptVQE terminated successfully with a final change in eigenvalue: %s", str(eigenvalue_diff))
                        termination_criterion = TerminationCriterion.CONVERGED
                        logger.debug("Reverting the addition of the last excitation to the ansatz since it resulted in a change of the eigenvalue below the configured threshold.")
                        self._excitation_list.pop()
                        theta.pop()
                        self._tmp_ansatz.operators = self._excitation_list
                        self.solver.ansatz = self._tmp_ansatz
                        self.solver.initial_point = theta
                        raw_vqe_result = prev_raw_vqe_result
                        break
                #Appending the computed eigenvalue to the tracking history
                history.append(raw_vqe_result.eigenvalue)
                logger.info("Current eigenvalue: %s", str(raw_vqe_result.eigenvalue))

                print(f"Iteration {iteration} done")
        else:
            # reached maximum number of iterations
            termination_criterion = TerminationCriterion.MAXIMUM
            logger.info("Maximum number of iterations reached. Finishing.")

        for angle in theta:
            self.optangles.append(angle)

        #Result setup
        result = AdaptVQEResult()
        result.num_iterations = iteration
        result.termination_criterion = termination_criterion
        result.eigenvalue_history = history

        if self.algorithm == 'adapt':
            if self.adapt1d is True:
                raw_vqe_result.optimal_parameters = self.dict_parameters
                raw_vqe_result.optimal_point = self.optangles
            result.combine(raw_vqe_result)
            result.final_max_gradient = max_grad[0]

            logger.info("The final eigenvalue is: %s", str(result.eigenvalue))
        elif self.algorithm == 'gga':
            if self.system == 'Molecule':
                logger.info("The final eigenvalue is: %s", str(result.eigenvalue_history[-1][0]))
            elif self.system == 'Ising':
                logger.info("The final eigenvalue is: %s", str(result.eigenvalue_history[-1]))            
        
        return self.indices, self.optangles, result, self.noiseless_energies, self.reoptimised_minimums, self.reoptimised_thetas

    def compute_noiseless_energy(self, theta, operator):
        '''
        Computes the energy in a perfect ideal situation, i.e. with no shot or real noise, of the system.
        Args:
            theta (list[float]): A list of optimal paramater values
            operator (BaseOperator): Operator (usually Hamiltonian) whose expectation value needs to be computed.
        Returns:
            the noiseless "ideal" energy of the system.
        '''

        job = self.estimator.run(self.solver.ansatz, operator, theta)
        results = job.result()

        return results.values[0].real

    def backward_optimization(self, theta, operator):
        '''
        Function to perform the backward optimization in the GGA-VQE framework.
        Consists of optimizing the N-1 parameters at the step N, each by a 1D local GGA optimization, from the N-1 to the 1st parameter.
        Args:  
            theta (list[float]): A list of (up to now) optimal parameter values
            operator (BaseOperator): Operator (usually Hamiltonian) whose expectation value is seek to be computed
        Returns:
            expected_minimum (float): The minimum energy after the backward optimization process
            theta (list[float]): A list of all the re-optimised parameters  
        '''

        if self.verbose:
            print("Starting reoptimisation step!!")

        iterations = len(self._excitation_list)-1
        
        reopti_minimums = []
        reopti_thetas = []
        for iteration in reversed(range(iterations)):
            tmp_ansatz = self.solver.ansatz.copy()
            data_points = []
            metadata = []

            for angle in self.spaced_thetas:
                theta[iteration] = angle
                if self.method == "circuit":
                    results = estimate_observables(self.solver.estimator, self.solver.ansatz, [operator], theta, shots=self.shots)
                elif self.method == 'evolution':
                    results = estimate_observables(self.solver.estimator, self._tmp_ansatz, [operator], theta, shots=self.shots)
                
                data_points.append(results[0][0])
                metadata.append(results[0][1])
            
            expected_minimum, theta_opt = uf.compute_minimum_in_direction(data_points, self.optimizer)
            theta[iteration] = theta_opt[0]
            if self.verbose:
                print("minimum after all re-optimization", expected_minimum)
                print("optimal angles after all re-optimization", theta)

            new_theta = theta.copy()
            new_theta[iteration] = theta_opt[0]
            reopti_minimums.append(expected_minimum)
            reopti_thetas.append(new_theta)
        
        self.reoptimised_minimums.append(reopti_minimums)
        self.reoptimised_thetas.append(reopti_thetas)
        
        return expected_minimum, theta

class AdaptVQEResult(VQEResult):
    """AdaptVQE Result."""

    def __init__(self) -> None:
        super().__init__()
        self._num_iterations: int | None = None
        self._final_max_gradient: float | None = None
        self._termination_criterion: str = ""
        self._eigenvalue_history: list[float] | None = None

    @property
    def num_iterations(self) -> int:
        """Returns the number of iterations."""
        return self._num_iterations

    @num_iterations.setter
    def num_iterations(self, value: int) -> None:
        """Sets the number of iterations."""
        self._num_iterations = value

    @property
    def final_max_gradient(self) -> float:
        """Returns the final maximum gradient."""
        return self._final_max_gradient

    @final_max_gradient.setter
    def final_max_gradient(self, value: float) -> None:
        """Sets the final maximum gradient."""
        self._final_max_gradient = value

    @property
    def termination_criterion(self) -> str:
        """Returns the termination criterion."""
        return self._termination_criterion

    @termination_criterion.setter
    def termination_criterion(self, value: str) -> None:
        """Sets the termination criterion."""
        self._termination_criterion = value

    @property
    def eigenvalue_history(self) -> list[float]:
        """Returns the history of computed eigenvalues.

        The history's length matches the number of iterations and includes the final computed value.
        """
        return self._eigenvalue_history

    @eigenvalue_history.setter
    def eigenvalue_history(self, eigenvalue_history: list[float]) -> None:
        """Sets the history of computed eigenvalues."""
        self._eigenvalue_history = eigenvalue_history