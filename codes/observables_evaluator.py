"""Evaluator of observables for algorithms."""

from __future__ import annotations
from collections.abc import Sequence
from typing import Any

import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import BaseEstimator
from qiskit.quantum_info.operators.base_operator import BaseOperator

from qiskit_algorithms.exceptions import AlgorithmError
from qiskit_algorithms.list_or_dict import ListOrDict

def estimate_observables(
    estimator: BaseEstimator,
    quantum_state: QuantumCircuit,
    observables: ListOrDict[BaseOperator],
    parameter_values: Sequence[float] | None = None,
    threshold: float = 1e-20,
    shots = None,
) -> ListOrDict[tuple[float, dict[str, Any]]]:
    """
    Args:
        estimator (BaseEstimator): An estimator primitive used for calculations.
        quantum_state (QuantumCircuit): A (parameterized) quantum circuit preparing a quantum state that expectation values are computed against.
        observables (list[BaseOperator]): A list of operators whose expectation values are to be calculated.
        parameter_values (list[float]): Optional list of parameters values to evaluate the quantum circuit on.
        threshold (float): A threshold value that defines which mean values should be neglected (helpful for ignoring numerical instabilities close to 0).
        shots (None or int): integer values of the number of shots that will be used for the simulation. (default is None, corresponding to 0 shots)
    Returns:
        A list or a dictionary of tuples (mean, metadata).

    Raises:
        AlgorithmError: If a primitive job is not successful.
    """
    if isinstance(observables, dict):
        observables_list = list(observables.values())
    else:
        observables_list = observables

    if len(observables_list) > 0:
        observables_list = _handle_zero_ops(observables_list)
        quantum_state = [quantum_state] * len(observables)
        parameter_values_: Sequence[float] | Sequence[Sequence[float]] | None = parameter_values
        if parameter_values is not None:
            parameter_values_ = [parameter_values] * len(observables)
        try:
            estimator_job = estimator.run(quantum_state, observables_list, parameter_values_, shots=shots)
            expectation_values = estimator_job.result().values
        except Exception as exc:
            raise AlgorithmError("The primitive job failed!") from exc

        metadata = estimator_job.result().metadata
        # Discard values below threshold
        observables_means = expectation_values * (np.abs(expectation_values) > threshold)
        # zip means and metadata into tuples
        observables_results = list(zip(observables_means, metadata))
    else:
        observables_results = []

    return _prepare_result(observables_results, observables)


def _handle_zero_ops(observables_list: list[BaseOperator]) -> list[BaseOperator]:
    """
    Replaces all occurrence of operators equal to 0 in the list with an equivalent SparsePauliOp operator.
    Args:
        observables_list (list[BaseOperator]): A list of operators whose expectation values are to be calculated.
    Returns:
        A list of operators made of only SparsePauliOp objects.
    """
    if observables_list:
        zero_op = SparsePauliOp.from_list([("I" * observables_list[0].num_qubits, 0)])
        for ind, observable in enumerate(observables_list):
            if observable == 0:
                observables_list[ind] = zero_op
    return observables_list

def _prepare_result(observables_results: list[tuple[float, dict]], observables: ListOrDict[BaseOperator]) -> ListOrDict[tuple[float, dict[str, Any]]]:
    """
    Prepares a list of tuples of eigenvalues and metadata tuples from observables_results and observables.
    Args:
        observables_results (list[tuple]): A list of tuples (mean, metadata).
        observables (list[BaseOperator]): A list of operators whose expectation values are to be calculated.
    Returns:
        A list or a dictionary of tuples (mean, metadata).
    """

    observables_eigenvalues: ListOrDict[tuple[float, dict]]

    if isinstance(observables, list):
        observables_eigenvalues = []
        for value in observables_results:
            observables_eigenvalues.append(value)

    else:
        observables_eigenvalues = {}
        for key, value in zip(observables.keys(), observables_results):
            observables_eigenvalues[key] = value

    return observables_eigenvalues