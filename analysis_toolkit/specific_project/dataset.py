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
"""Dataset module for this analysis"""


import dataclasses
import os
import re
from itertools import combinations
from typing import Generator

import nibabel as nib
import numpy as np
import pandas as pd
import yaml

from analysis_toolkit.basic_toolkit.dataset import Database as _Database

from .types import (
    Callable,
    _CenterConfig,
    _Hemisphere,
    _LocationConfig,
    _Region,
    _RegionConfig,
    _RegionInfo,
    _Stimulation,
    _Structure,
)


def get_region(structure: _Structure, hemisphere: _Hemisphere) -> _Region:
    return f"{structure} {hemisphere}"


def get_structure(region: _Region) -> _Structure:
    return region.split()[0]


def get_hemisphere(region: _Region) -> _Hemisphere:
    return region.split()[1]


class Location:
    FULL_SHAPE = (53, 63, 46)

    def __new__(cls, path: str) -> pd.DataFrame:
        with open(path, "r") as file:
            config: _LocationConfig = yaml.safe_load(file)
        location_df = pd.DataFrame(columns=_RegionInfo.__annotations__.keys())
        for region_info in cls.iter_config(config):

            region_df = pd.Series(region_info).to_frame().T
            location_df = pd.concat(
                [location_df, region_df], ignore_index=True
            )
        location_df.sort_values(by="region", inplace=True)
        location_df.reset_index(drop=True, inplace=True)
        return location_df

    @classmethod
    def iter_config(
        cls, config: _LocationConfig
    ) -> Generator[_RegionInfo, None, None]:
        shift: int = config["Shift"]
        center_config: _CenterConfig = config["Center"]
        for structure, structure_config in center_config.items():
            for hemisphere, region_config in structure_config.items():
                region_info = cls.get_region_info(
                    structure, hemisphere, region_config, shift
                )
                yield region_info

    @classmethod
    def get_region_info(
        cls,
        structure: _Structure,
        hemisphere: _Hemisphere,
        region_config: _RegionConfig,
        shift: int,
    ) -> _RegionInfo:
        region = get_region(structure, hemisphere)
        x = region_config["x"]
        y = region_config["y"]
        z = region_config["z"]
        x_slice = cls.get_slice(x, shift)
        y_slice = cls.get_slice(y, shift)
        z_slice = cls.get_slice(z, shift)
        return _RegionInfo(
            region=region,
            structure=structure,
            hemisphere=hemisphere,
            x=x,
            y=y,
            z=z,
            shift=shift,
            x_slice=x_slice,
            y_slice=y_slice,
            z_slice=z_slice,
        )

    @staticmethod
    def get_slice(center: int, shift: int) -> slice:
        return slice(center - shift, center + shift)


class Database:
    STIMULATIONS = ["Auditory", "Visual"]

    def __new__(cls, dirname: str) -> pd.DataFrame:
        database = []
        for stimulation in cls.STIMULATIONS:
            stim_dirname = os.path.join(dirname, stimulation)
            subjects = os.listdir(stim_dirname)
            subjects = sorted(
                subjects, key=lambda x: int(re.findall(r"\d+$", x)[0])
            )
            subjects_data_path = cls.get_subjects_data_path(
                dirname=stim_dirname,
                stimulation=stimulation,
                subjects=subjects,
            )
            database.append(
                _Database(
                    path_lst=subjects_data_path,
                    stimulation=stimulation,
                    subject_lst=subjects,
                    gen_random=True,
                )
            )
        return pd.concat(database, ignore_index=True)

    @classmethod
    def get_subjects_data_path(
        cls, dirname: str, stimulation: _Stimulation, subjects: list[str]
    ) -> list[str]:
        match stimulation:
            case "Auditory":
                return cls._get_auditory_data_path(
                    dirname=dirname, subjects=subjects
                )
            case "Visual":
                return cls._get_visual_data_path(
                    dirname=dirname, subjects=subjects
                )
            case _:
                raise ValueError(f"Invalid stimulation type: {stimulation}")

    @classmethod
    def _get_auditory_data_path(
        cls, dirname: str, subjects: list[str]
    ) -> list[str]:
        def get_path(dirname: str, subject: str) -> str:
            subject_num = re.findall(r"Subject_(\d+)", subject)[0]
            subject_num = int(subject_num)
            filename = f"Words_{subject_num}.nii"
            return os.path.join(dirname, subject, filename)

        return cls._get_data_path(
            dirname=dirname, subjects=subjects, get_path_func=get_path
        )

    @classmethod
    def _get_visual_data_path(
        cls, dirname: str, subjects: list[str]
    ) -> list[tuple[str, str]]:
        def get_path(dirname: str, subject: str) -> str:
            return os.path.join(dirname, subject, "con_0006.img")

        return cls._get_data_path(
            dirname=dirname, subjects=subjects, get_path_func=get_path
        )

    @staticmethod
    def _get_data_path(
        dirname: str, subjects: list[str], get_path_func: Callable
    ) -> list[str]:
        data_path = []
        for subject in subjects:
            path = get_path_func(dirname=dirname, subject=subject)
            data_path.append(path)
        return data_path


class Dataset:
    def __new__(cls, dirname: str) -> pd.DataFrame:
        location_path = os.path.join(dirname, "location.yaml")
        location = Location(path=location_path)
        database = Database(dirname=dirname)

        dataset_lst = []
        for _, row_location in location.iterrows():
            region = row_location["region"]
            structure = row_location["structure"]
            hemisphere = row_location["hemisphere"]
            x_slice = row_location["x_slice"]
            y_slice = row_location["y_slice"]
            z_slice = row_location["z_slice"]
            for _, row_database in database.iterrows():
                data_type = row_database["data_type"]
                stimulation = row_database["stimulation"]
                subject = row_database["subject"]
                data = row_database["data"]
                data = data[x_slice, y_slice, z_slice]
                dataset = (
                    pd.Series(
                        {
                            "region": region,
                            "structure": structure,
                            "hemisphere": hemisphere,
                            "data_type": data_type,
                            "stimulation": stimulation,
                            "subject": subject,
                            "data": data,
                        }
                    )
                    .to_frame()
                    .T
                )
                dataset_lst.append(dataset)
        return pd.concat(dataset_lst, ignore_index=True)


class PatternDataset:
    def __new__(
        cls,
        dataset: pd.DataFrame,
        stimulation: _Stimulation,
        n_subject: int,
        include_self: bool = False,
    ) -> pd.DataFrame:
        dataset = dataset[dataset["data_type"] == "Real"]
        dataset = dataset[dataset["stimulation"] == stimulation]
        subjects = dataset["subject"].unique()
        pattern_dataset_lst = []
        for subject in subjects:
            if include_self:
                subject_dataset = dataset.copy()
            else:
                subject_dataset = dataset[dataset["subject"] != subject]
            regions = subject_dataset["region"].unique()
            for region in regions:
                structure = get_structure(region)
                hemisphere = get_hemisphere(region)
                region_dataset: pd.DataFrame = subject_dataset[
                    subject_dataset["region"] == region
                ]
                region_dataset_idx = region_dataset.index.to_list()
                for pattern_idx, pattern_subject_idx in enumerate(
                    combinations(region_dataset_idx, n_subject)
                ):
                    pattern_dataset = region_dataset.loc[pattern_subject_idx]
                    pattern_data = np.mean(pattern_dataset["data"])
                    pattern_data = (
                        pd.Series(
                            {
                                "region": region,
                                "structure": structure,
                                "hemisphere": hemisphere,
                                "data_type": "Pattern",
                                "stimulation": stimulation,
                                "subject": pattern_idx,
                                "data": pattern_data,
                            }
                        )
                        .to_frame()
                        .T
                    )
                    pattern_dataset_lst.append(pattern_data)
        return pd.concat(pattern_dataset_lst, ignore_index=True)


class SimilarityDataset:
    pass
    # TODO: Add the SimilarityDataset class
    # matrix, region, structure, hemisphere, modify_method, similarity, random_visual, random_auditory, real_visual, real_auditory, pattern_visual, pattern_auditory
    # e.g. random_visual: can be None of list of idx. If is random or
    #   real, it need to be idx lst of dataset. If is pattern, it need
    #   to be idx lst of pattern dataset. pattern idx need to along the
    #   axis 1 of the matrix.


# @dataclasses.dataclass
# class SubjectSimilarityVector:
#     subject_idx: int
#     data_type: _DATA_TYPE
#     stimulation: _STIMULATION
#     structure: _STRUCTURE
#     hemisphere: _HEMISPHERE
#     subject: str
#     idx: int
#     similarity_vector: np.ndarray


# @dataclasses.dataclass
# class SimilarityMatrix:
#     matrix: np.ndarray
#     region: _REGION
#     structure: _STRUCTURE
#     hemisphere: _HEMISPHERE
#     stimulation_slices: dict[
#         Literal[
#             "Random Auditory", "Random Visual", "Real Auditory", "Real Visual"
#         ],
#         slice,
#     ]
#     subject_vectors: dict[int, SubjectSimilarityVector] | None = None
