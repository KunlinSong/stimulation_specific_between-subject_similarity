#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Kunlin SONG"
__copyright__ = "Copyright (c) 2024 Kunlin SONG"
__license__ = "MIT"
__email__ = "kunlinsongcode@gmail.com"


from typing import Callable, Literal, NamedTuple, Optional, Union, overload

import numpy as np
from scipy import stats

__all__ = ["BootstrapTest"]


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
        >>> result = test(data, confidence_level="*")
    """

    RANDOM_SEED = 42
    CONFIDENCE_METHOD = "BCa"

    def __init__(self, statistic: Callable = np.mean, n_resamples: int = 10000) -> None:
        self.statistic = statistic
        self.n_resamples = n_resamples

    @overload
    def __call__(
        self, data: np.ndarray, *, confidence_level: Literal["*", "**", "***"]
    ) -> BootstrapTestResult: ...

    @overload
    def __call__(
        self, data: np.ndarray, *, confidence_levels: list[Literal["*", "**", "***"]]
    ) -> BootstrapTestResult: ...

    def __call__(
        self,
        data: np.ndarray,
        *,
        confidence_level: Optional[Literal["*", "**", "***"]] = None,
        confidence_levels: Optional[list[Literal["*", "**", "***"]]] = None,
    ) -> Union[BootstrapTestResult, BootstrapTestResult]:
        """
        Perform bootstrap hypothesis testing.

        Args:
            data (np.ndarray): The input data.
            confidence_level (Literal["*", "**", "***"], optional): The
            confidence level for the test. Default is None.
            confidence_levels (list[Literal["*", "**", "***"]], optional):
            The list of confidence levels for the test. Default is None.

        Returns:
            BootstrapTestResult or BootstrapTestResults: The result of
                the bootstrap test.
        """
        if confidence_level is not None:
            confidence_levels = [confidence_level]
        confidence_intervals = {}
        statistic = None
        for confidence_level in confidence_levels:
            match confidence_level:
                case "*":
                    confidence_num = 0.95
                case "**":
                    confidence_num = 0.99
                case "***":
                    confidence_num = 0.999
                case _:
                    raise ValueError(
                        "Invalid confidence level. Please choose from \
                        '*', '**', or '***'."
                    )
            res = stats.bootstrap(
                data=data,
                statistic=self.statistic,
                n_resamples=self.n_resamples,
                confidence_level=confidence_num,
                method=self.CONFIDENCE_METHOD,
                random_state=self.RANDOM_SEED,
            )
            if statistic is None:
                statistic = np.median(res.bootstrap_distribution)
                distribution = res.bootstrap_distribution
            confidence_interval = ConfidenceInterval(
                low=res.confidence_interval.low, high=res.confidence_interval.high
            )
            confidence_intervals[confidence_level] = confidence_interval
        return BootstrapTestResult(
            statistic=statistic,
            distribution=distribution,
            confidence_interval=confidence_intervals,
        )
