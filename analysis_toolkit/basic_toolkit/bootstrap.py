"""A module for bootstrap testing."""

from dataclasses import dataclass

import numpy as np
from scipy import stats

from analysis_toolkit.basic_toolkit.types import Callable

__all__ = ["BootstrapTest"]


@dataclass
class _ConfidenceInterval:
    """A dataclass for confidence interval.

    Attributes:
        low: The lower bound of the confidence interval.
        high: The upper bound of the confidence interval.
    """

    low: float
    high: float


@dataclass
class _BootstrapTestResult:
    """A dataclass for the result of bootstrap testing.

    Attributes:
        statistic: The value of the statistic.
        distribution: The distribution of the bootstrap resamples.
        confidence_interval: The confidence interval for the statistic.
    """

    statistic: float
    distribution: np.ndarray
    confidence_interval: _ConfidenceInterval


class BootstrapTest:
    """A class that performs bootstrap testing.

    Using the bootstrap method to test the hypothesis of the statistic,
    and returns the result of the test.  The random seed for bootstrap is
    fixed to 42.  The confidence method is BCa.

    Attributes:
        RANDOM_SEED: The random seed for bootstrap.  Default is 42.
        CONFIDENCE_METHOD: The confidence method.  Default is "BCa".
        CONFIDENCE_LEVEL: The confidence level.  Default is 0.95
    """

    RANDOM_SEED = 42
    CONFIDENCE_METHOD = "BCa"
    CONFIDENCE_LEVEL = 0.95

    def __init__(self, statistic: Callable, n_resamples: int = 10000) -> None:
        """Initializes the instance based on the function to compute
        the statistic and the number of resamples.

        Args:
            statistic: The function to compute the statistic.
            n_resamples: The number of resamples. Default is 10000.
        """
        self.statistic = statistic
        self.n_resamples = n_resamples

    def __call__(self, data: np.ndarray) -> _BootstrapTestResult:
        """Performs the bootstrap test on the data.

        Args:
            data: The input data.

        Returns:
            The result of bootstrap testing.
        """
        res = stats.bootstrap(
            data,
            self.statistic,
            n_resamples=self.n_resamples,
            confidence_level=self.CONFIDENCE_LEVEL,
            method=self.CONFIDENCE_METHOD,
            random_seed=self.RANDOM_SEED,
        )
        return _BootstrapTestResult(
            statistic=np.median(res[1]),
            distribution=res[1],
            confidence_interval=_ConfidenceInterval(
                low=res[0][0], high=res[0][1]
            ),
        )
