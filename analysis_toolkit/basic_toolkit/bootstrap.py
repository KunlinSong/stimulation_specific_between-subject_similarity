# MIT License
#
# Copyright (c) 2024 Kunlin SONG
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""A module for bootstrap testing."""


import numpy as np
from scipy import stats

from analysis_toolkit.basic_toolkit.types import (
    Callable,
    _BootstrapTestResult,
    _ConfidenceInterval,
    _ConfidenceIntervalDict,
)

__all__ = ["BootstrapTest"]


class BootstrapTest:
    """A class that performs bootstrap testing.

    Using the bootstrap method to test the hypothesis of the statistic,
    and returns the result of the test.  The random seed for bootstrap is
    fixed to 42.  The confidence method is BCa.

    Attributes:
        RANDOM_SEED: The random seed for bootstrap. Default is 42.
        CONFIDENCE_METHOD: The confidence method. Default is "BCa".
    """

    RANDOM_SEED = 42
    CONFIDENCE_METHOD = "BCa"

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
        statistics = None
        distribution = None
        confidence_interval_dict = {}

        for idx, confidence_level in enumerate([0.95, 0.99, 0.999, 0.9999]):
            res = stats.bootstrap(
                data,
                self.statistic,
                n_resamples=self.n_resamples,
                confidence_level=confidence_level,
                method=self.CONFIDENCE_METHOD,
                random_seed=self.RANDOM_SEED,
            )
            if idx == 0:
                statistics = np.median(res[1])
                distribution = res[1]
            confidence_interval_dict[idx] = _ConfidenceInterval(
                low=res[0][0], high=res[0][1]
            )
        return _BootstrapTestResult(
            statistic=statistics,
            distribution=distribution,
            confidence_intervals=_ConfidenceIntervalDict(
                one_star=confidence_interval_dict[0],
                two_stars=confidence_interval_dict[1],
                three_stars=confidence_interval_dict[2],
            ),
        )
