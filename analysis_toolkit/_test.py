from typing import Callable, NamedTuple, Optional, Union, overload

import numpy as np
from scipy import stats

ConfidenceInterval = NamedTuple(
    "ConfidenceInterval",
    [("low", float), ("high", float)],
)


BootstrapTestResult = NamedTuple(
    "BootstrapTestResultList",
    [
        ("statistic", float),
        ("distribution", np.ndarray),
        ("confidence_interval", dict[float, ConfidenceInterval]),
    ],
)


class BootstrapTest:
    """
    A class that performs bootstrap hypothesis testing.

    Args:
        statistic (Callable): The statistic to be computed on resampled
            data. Default is np.mean.
        n_resamples (int): The number of resamples to generate. Default
            is 10000.

    Returns:
        BootstrapTestResult or BootstrapTestResults: The result of the
            bootstrap test.

    Examples:
        >>> test = BootstrapTest()
        >>> result = test(data, confidence_level=0.95)
    """

    RANDOM_SEED = 42
    CONFIDENCE_METHOD = "BCa"

    def __init__(self, statistic: Callable = np.mean, n_resamples: int = 10000) -> None:
        self.statistic = statistic
        self.n_resamples = n_resamples

    @overload
    def __call__(
        self, data: np.ndarray, *, confidence_level: float
    ) -> BootstrapTestResult: ...

    @overload
    def __call__(
        self, data: np.ndarray, *, confidence_levels: list[float]
    ) -> BootstrapTestResult: ...

    def __call__(
        self,
        data: np.ndarray,
        *,
        confidence_level: Optional[float] = None,
        confidence_levels: Optional[list[float]] = None,
    ) -> Union[BootstrapTestResult, BootstrapTestResult]:
        """
        Perform bootstrap hypothesis testing.

        Args:
            data (np.ndarray): The input data.
            confidence_level (float, optional): The confidence level for
                the test. Default is None.
            confidence_levels (list[float], optional): The list of
                confidence levels for the test. Default is None.

        Returns:
            BootstrapTestResult or BootstrapTestResults: The result of
                the bootstrap test.
        """
        if confidence_level is not None:
            confidence_levels = [confidence_level]
        confidence_intervals = {}
        statistic = None
        for confidence_level in confidence_levels:
            res = stats.bootstrap(
                data=data,
                statistic=self.statistic,
                n_resamples=self.n_resamples,
                confidence_level=confidence_level,
                method=self.CONFIDENCE_METHOD,
                random_state=self.RANDOM_SEED,
            )
            if statistic is None:
                statistic = np.median(res.bootstrap_distribution)
                distribution = res.bootstrap_distribution
            confidence_interval = ConfidenceInterval(
                low=res.confidence_interval.low, high=res.confidence_interval.high
            )
            confidence_intervals[f"{confidence_level}"] = confidence_interval
        return BootstrapTestResult(
            statistic=statistic,
            distribution=distribution,
            confidence_interval=confidence_intervals,
        )
