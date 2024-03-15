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
"""A basic toolkit for dataset."""


import os

import numpy as np
import pandas as pd
from nibabel.loadsave import load as nib_load

from ._types import Any, Stimulation

__all__ = [
    "Database",
    "RandomDatabase",
    "RealDatabase",
]


class _RealSubjectData:
    """A class to get a pandas DataFrame from a path.

    Attributes:
        DATA_TYPE: The data type label for the real data.
    """

    DATA_TYPE = "Real"

    def __new__(
        cls,
        path: str,
        stimulation: Stimulation,
        subject: str | None = None,
    ) -> pd.DataFrame:
        """Create a pandas DataFrame from a path.

        We use path to get the fMRI data and then create a pandas
        DataFrame with data_type, stimulation, subject, and data as
        columns.

        Args:
            path: The path to the fMRI data.
            stimulation: The stimulation type.
            subject: The subject name.  A optional argument to
              distinguish different receivers of the same stimulation.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path {path} not found.")
        if not os.path.isfile(path):
            raise ValueError(f"Path {path} is not a file.")
        data = cls._get_data(path)
        subject_data = {
            "data_type": cls.DATA_TYPE,
            "stimulation": stimulation,
            "subject": subject,
            "data": data,
        }
        subject_data = {
            k: cls._value_to_series(v) for k, v in subject_data.items()
        }
        return pd.DataFrame(subject_data)

    @staticmethod
    def _value_to_series(value: Any) -> pd.Series:
        return pd.Series(
            data=value,
        )

    @classmethod
    def _get_data(cls, path: str) -> np.ndarray:
        data = np.array(nib_load(path).get_fdata())
        return data


class RealDatabase:
    """A class to get a pandas DataFrame from a list of paths."""

    def __new__(
        cls,
        paths_lst: list[str],
        stimulation,
        subject_lst: list[str] | None = None,
    ) -> pd.DataFrame:
        """Create a pandas DataFrame from a list of paths.

        Args:
            paths_lst: A list of paths to the fMRI data.
            stimulation: The stimulation type.
            subject_lst: A list of subject names with the same length of
              paths_lst.  An optional argument to distinguish different
              receivers of the same stimulation.
        """
        if (n_path := len(paths_lst)) == 0:
            raise ValueError("Path list is empty.")
        elif n_path == 1:
            return _RealSubjectData(
                path=paths_lst[0],
                stimulation=stimulation,
                subject=subject_lst[0] if subject_lst else None,
            )
        else:
            data = []
            for path_idx, path in enumerate(paths_lst):
                data.append(
                    _RealSubjectData(
                        path=path,
                        stimulation=stimulation,
                        subject=subject_lst[path_idx] if subject_lst else None,
                    )
                )
            return pd.concat(data, ignore_index=True)


class RandomDatabase:
    """A class to get a pandas DataFrame with random brain images.

    We use the real database to get the shape of the brain images and
    then randomize the real data by shuffling the non-NaN values to
    create a random database.

    Attributes:
        RANDOM_SEED: The random seed to shuffle the non-NaN values.
        DATA_TYPE: The data type label for the random data.
    """

    RANDOM_SEED = 42
    DATA_TYPE = "Random"

    def __new__(cls, real_database: pd.DataFrame) -> pd.DataFrame:
        """Create a pandas DataFrame with random brain images.

        Args:
            real_database: The real database which is used to create the
              random database.
        """
        data = real_database.copy()
        data["data_type"] = cls.DATA_TYPE
        data["data"] = data["data"].apply(func=cls._randomize)
        return data

    @classmethod
    def _randomize(cls, data: np.ndarray) -> np.ndarray:
        data = data.copy()
        non_nan_indices = np.nonzero(~np.isnan(data))
        random_generator = np.random.default_rng(cls.RANDOM_SEED)
        random_indices = cls._shuffle_indices(
            indices=non_nan_indices, random_generator=random_generator
        )
        data[non_nan_indices] = data[random_indices]
        return data

    @staticmethod
    def _shuffle_indices(
        indices: tuple[np.ndarray], random_generator: np.random.Generator
    ) -> tuple[np.ndarray]:
        random_indices = np.stack(indices, axis=0)
        random_generator.shuffle(random_indices, axis=1)
        random_indices = tuple(random_indices)
        return random_indices


class Database:
    """A class to get a pandas DataFrame from a list of paths."""

    def __new__(
        cls,
        path_lst: list[str],
        stimulation: Stimulation,
        gen_random: bool = True,
    ) -> pd.DataFrame:
        """Create a pandas DataFrame from a list of paths.

        Args:
            path_lst: A list of paths to the fMRI data.
            stimulation: The stimulation type.
            gen_random: Generate random data and concatenate it with the
              real data if True.  Default is True.
        """
        database = RealDatabase(paths_lst=path_lst, stimulation=stimulation)
        if gen_random:
            random_database = RandomDatabase(real_database=database)
            database = pd.concat(
                [database, random_database], ignore_index=True
            )
        return database
