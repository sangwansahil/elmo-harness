"""Closed-loop machinery — diagnose, regression suite, deployment gate."""

from .diagnose import FailureCluster, diagnose
from .gate import GateResult, capability_vector, evaluate_gate
from .regression import RegressionSuite, RegressionCase

__all__ = [
    "FailureCluster",
    "diagnose",
    "GateResult",
    "capability_vector",
    "evaluate_gate",
    "RegressionSuite",
    "RegressionCase",
]
