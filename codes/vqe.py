"""The variational quantum eigensolver algorithm."""

from __future__ import annotations

import logging
from time import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.primitives import BaseEstimator
from qiskit.quantum_info.operators.base_operator import BaseOperator

from qiskit_algorithms.exceptions import AlgorithmError
from qiskit_algorithms.optimizers import Optimizer, Minimizer, OptimizerResult
from qiskit_algorithms.variational_algorithm import VariationalAlgorithm, VariationalResult
from qiskit_algorithms.minimum_eigensolvers import MinimumEigensolver, MinimumEigensolverResult
from qiskit_algorithms.utils import validate_initial_point, validate_bounds

logger = logging.getLogger(__name__)

class VQE(VariationalAlgorithm, MinimumEigensolver):
    """
    The following attributes can be set via the initializer but can also be read and updated once
    the VQE object has been constructed.

    Attributes:
        estimator (BaseEstimator): The estimator primitive to compute the expectation value of the Hamiltonian operator.
        ansatz (QuantumCircuit): A parameterized quantum circuit to prepare the trial state.
        optimizer (Optimizer | Minimizer): A classical optimizer to find the minimum energy. This can either be a Qiskit optimizer or a callable implementing the minimizer protocol.
        callback (Callable[[int, np.ndarray, float, dict[str, Any]], None] | None): A callback that can access the intermediate data at each optimization step. 
        These data are: the evaluation count, the optimizer parameters for the ansatz, the evaluated mean, and the metadata dictionary.
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        ansatz: QuantumCircuit,
        optimizer: Optimizer | Minimizer,
        *,
        initial_point: Sequence[float] | None = None,
        callback: Callable[[int, np.ndarray, float, dict[str, Any]], None] | None = None,
        shots: int | None = None,
        dict_initial_point: dict[str, float] | None = None,
    ) -> None:

        super().__init__()

        self.estimator = estimator
        self.ansatz = ansatz
        self.optimizer = optimizer
        self.callback = callback
        self.dict_initial_point = dict_initial_point

        #Defined through a property decorater because the VariationalAlgorithm interface needs getter and setter
        self.initial_point = initial_point

        #Adding possibility to use shot noise when evaluating the cost function
        self.shots = shots

    @property
    def initial_point(self) -> Sequence[float] | None:
        return self._initial_point

    @initial_point.setter
    def initial_point(self, value: Sequence[float] | None) -> None:
        self._initial_point = value

    def compute_minimum_eigenvalue(self, operator: BaseOperator) -> VQEResult:
        self._check_operator_ansatz(operator)

        bounds = validate_bounds(self.ansatz)

        if self.dict_initial_point is not None:
            self.ansatz.assign_parameters(self.dict_initial_point, inplace=True)
            initial_point = self.initial_point
        else:
            initial_point = validate_initial_point(self.initial_point, self.ansatz)

        start_time = time()

        evaluate_energy = self._get_evaluate_energy(self.ansatz, operator)

        # perform optimization with distinction between full or local method
        if self.dict_initial_point is not None:
            optimizer_result = self.optimizer.minimize(fun=evaluate_energy, x0=initial_point[-1], bounds=bounds)
        else:
            optimizer_result = self.optimizer.minimize(fun=evaluate_energy, x0=initial_point, bounds=bounds)

        optimizer_time = time() - start_time

        logger.info("Optimization complete in %s seconds.\n Found optimal point %s", optimizer_time, optimizer_result.x)

        return self._build_vqe_result(self.ansatz, optimizer_result, optimizer_time)

    def _get_evaluate_energy(self, ansatz: QuantumCircuit, operator: BaseOperator) -> Callable[[np.ndarray], np.ndarray | float]:
        """
        Args:
            ansatz: The ansatz preparing the quantum state.
            operator: The operator whose energy to evaluate.

        Returns:
            A callable that computes and returns the energy of the hamiltonian of each parameter.

        Raises:
            AlgorithmError: If failure of the primitive job to evaluate the energy.
        """
        num_parameters = ansatz.num_parameters

        # avoid creating an instance variable to remain stateless regarding results
        eval_count = 0

        def evaluate_energy(parameters: np.ndarray) -> np.ndarray | float:
            nonlocal eval_count

            # Ensure parameters is of shape [array, array, ...]
            parameters = np.reshape(parameters, (-1, num_parameters)).tolist()
            batch_size = len(parameters)

            try:
                job = self.estimator.run(batch_size * [ansatz], batch_size * [operator], parameters, shots=self.shots)
                estimator_result = job.result()
            except Exception as exc:
                raise AlgorithmError("The primitive job to evaluate the energy failed!") from exc

            values = estimator_result.values

            if self.callback is not None:
                metadata = estimator_result.metadata
                for params, value, meta in zip(parameters, values, metadata):
                    eval_count += 1
                    self.callback(eval_count, params, value, meta)

            if len(values) == 1:
                energy = values[0]
            else:
                energy = values

            return energy

        return evaluate_energy

    def _check_operator_ansatz(self, operator: BaseOperator):
        if operator.num_qubits != self.ansatz.num_qubits:
            try:
                logger.info("Trying to resize ansatz to match operator on %s qubits.", operator.num_qubits)
                print("Trying to resize ansatz to match operator on %s qubits...", operator.num_qubits)
                self.ansatz.num_qubits = operator.num_qubits
            except AttributeError as error:
                raise AlgorithmError("...FAILED!!! \n The number of qubits of the ansatz does not match the operator, and the ansatz does not allow setting the number of qubits.") from error
            print("...SUCCESS!!!")
        if self.ansatz.num_parameters == 0:
            raise AlgorithmError("The parametrized ansatz has no free parameters!! \n CANCELLING CALCULATIONS")

    def _build_vqe_result(self, ansatz: QuantumCircuit, optimizer_result: OptimizerResult, optimizer_time: float) -> VQEResult:
        result = VQEResult()
        result.optimal_circuit = ansatz.copy()
        result.eigenvalue = optimizer_result.fun
        result.cost_function_evals = optimizer_result.nfev
        result.optimal_point = optimizer_result.x
        result.optimal_parameters = dict(zip(self.ansatz.parameters, optimizer_result.x))
        result.optimal_value = optimizer_result.fun
        result.optimizer_time = optimizer_time
        result.optimizer_result = optimizer_result
        return result

class VQEResult(VariationalResult, MinimumEigensolverResult):
    """The Variational Quantum Eigensolver (VQE) result."""

    def __init__(self) -> None:
        super().__init__()
        self._cost_function_evals: int | None = None

    @property
    def cost_function_evals(self) -> int | None:
        """The number of cost optimizer evaluations."""
        return self._cost_function_evals

    @cost_function_evals.setter
    def cost_function_evals(self, value: int) -> None:
        self._cost_function_evals = value