"""Dataset module for this specific project."""

import dataclasses
import os
import re
from itertools import combinations
from typing import Generator, Literal

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

__all__ = [
    "add_similarity_data",
    "get_region",
    "get_structure",
    "get_hemisphere",
    "Dataset",
    "PatternDataset",
]


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


@dataclasses.dataclass
class IndexInfo:
    subject_idx: int
    data_type: Literal["Random", "Real", "Pattern"]
    stimulation: _Stimulation


def add_similarity_data(
    similarity_dataset: pd.DataFrame,
    matrix: np.ndarray,
    region: str,
    structure: str,
    hemisphere: str,
    modify_method: str,
    similarity: str,
    matrix_type: Literal["subject-subject", "subject-pattern"],
    x_index_info: dict[int, IndexInfo],
    y_index_info: dict[int, IndexInfo],
) -> pd.DataFrame:
    """Append a new similarity data to the similarity dataset.

    Args:
        similarity_dataset: The dataset to append the new similarity data.
        matrix: The similarity matrix.
        region: The region of the brain.
        structure: The structure of the brain.
        hemisphere: The hemisphere of the brain.
        modify_method: The preprocessing method used before calculating
          the similarity.
        similarity: The similarity method used to calculate the similarity.
        x_index_info: A dictionary of the similarity's index's
          information of the x-axis of the matrix.
        y_index_info: A dictionary of the similarity's index's
          information of the y-axis of the matrix.

    Returns:
        The similarity dataset with the new similarity data.
    """
    data = (
        pd.Series(
            {
                "matrix": matrix,
                "region": region,
                "structure": structure,
                "hemisphere": hemisphere,
                "modify_method": modify_method,
                "similarity": similarity,
                "matrix_type": matrix_type,
                "x_indices": x_index_info,
                "y_indices": y_index_info,
            }
        )
        .to_frame()
        .T
    )
    similarity_dataset = pd.concat(
        [similarity_dataset, data], ignore_index=True
    )
    return similarity_dataset
