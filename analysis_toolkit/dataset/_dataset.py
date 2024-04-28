import os
import re
from dataclasses import dataclass
from functools import partial
from itertools import combinations
from tkinter import W
from typing import Callable, Generator, Literal, TypedDict

import nibabel
import numpy as np
import pandas as pd
import yaml

__all__ = [
    "BrainRegionsDataFrame",
    "IndexInfo",
    "PatternDataset",
    "WholeBrainDataFrame",
    "add_similarity_data",
    "get_hemisphere",
    "get_region",
    "get_structure",
]


class _RegionConfig(TypedDict):
    x: int
    y: int
    z: int


class _StructureConfig(TypedDict):
    L: _RegionConfig
    R: _RegionConfig


class _CenterConfig(TypedDict):
    FFA: _StructureConfig
    STG: _StructureConfig


class _LocationConfig(TypedDict):
    Shift: int
    Center: _CenterConfig


_Hemisphere = Literal["L", "R"]
_Structure = Literal["FFA", "STG"]
_Region = Literal["FFA L", "FFA R", "STG L", "STG R"]


class _RegionInfo(TypedDict):
    region: _Region
    structure: _Structure
    hemisphere: _Hemisphere
    x: int
    y: int
    z: int
    shift: int
    x_slice: slice
    y_slice: slice
    z_slice: slice


_DataType = Literal["Random", "Real"]
_Stimulation = Literal["Auditory", "Visual"]
_PreprocessMethod = Literal["FFT", "Gradient", "Spatial Average"]


def get_region(structure: _Structure, hemisphere: _Hemisphere) -> _Region:
    return f"{structure} {hemisphere}"


def get_structure(region: _Region) -> _Structure:
    return region.split()[0]


def get_hemisphere(region: _Region) -> _Hemisphere:
    return region.split()[1]


def _get_real_subject_data(
    path: str, stimulation: _Stimulation, subject: str | None
) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path {path} not found.")
    if not os.path.isfile(path):
        raise ValueError(f"Path {path} is not a file.")
    data = np.array(nibabel.load(path).get_fdata())
    subject_data = {
        "data_type": "Real",
        "stimulation": stimulation,
        "subject": subject,
        "data": data,
    }
    return pd.Series(subject_data).to_frame().T


def _batch_get_real_subject_data(
    paths_lst: list[str],
    stimulation: _Stimulation,
    subject_lst: list[str] | None = None,
) -> pd.DataFrame:
    if (n_path := len(paths_lst)) == 0:
        raise ValueError("Path list is empty.")
    elif n_path == 1:
        return _get_real_subject_data(
            path=paths_lst[0],
            stimulation=stimulation,
            subject=subject_lst[0] if subject_lst else None,
        )
    else:
        data_lst = []
        get_real_subject_data = partial(
            _get_real_subject_data, stimulation=stimulation
        )
        for path_idx, path in enumerate(paths_lst):
            data_lst.append(
                get_real_subject_data(
                    path=path,
                    subject=subject_lst[path_idx] if subject_lst else None,
                )
            )
        return pd.concat(data_lst, ignore_index=True)


def _randomize_subject_data(
    data: pd.DataFrame, random_seed: int = 42
) -> pd.DataFrame:
    data = data.copy()
    non_nan_indices = np.nonzero(~np.isnan(data))
    random_generator = np.random.default_rng(random_seed)
    random_indices = np.stack(non_nan_indices, axis=0)
    random_generator.shuffle(random_indices, axis=-1)
    random_indices = tuple(random_indices)
    data[non_nan_indices] = data[random_indices]
    return data


def _get_random_subject_dataframe(
    real_dataframe: pd.DataFrame,
    random_seed: int = 42,
) -> pd.DataFrame:
    random_dataframe = real_dataframe.copy()
    random_dataframe["data_type"] = "Random"
    random_dataframe["data"] = random_dataframe["data"].apply(
        partial(_randomize_subject_data, random_seed=random_seed)
    )
    return random_dataframe


class LocationDataFrame(pd.DataFrame):
    FULL_SHAPE = (53, 63, 46)

    @classmethod
    def load_from_location_config(cls, path: str) -> pd.DataFrame:
        with open(path, "r") as file:
            config: _LocationConfig = yaml.safe_load(file)
        location_df = pd.DataFrame(columns=_RegionInfo.__annotations__.keys())
        for region_info in cls._iter_config(config):

            region_df = pd.Series(region_info).to_frame().T
            location_df = pd.concat(
                [location_df, region_df], ignore_index=True
            )
        location_df.sort_values(by="region", inplace=True)
        location_df.reset_index(drop=True, inplace=True)
        return location_df

    @classmethod
    def _iter_config(
        cls, config: _LocationConfig
    ) -> Generator[_RegionInfo, None, None]:
        shift: int = config["Shift"]
        center_config: _CenterConfig = config["Center"]
        for structure, structure_config in center_config.items():
            for hemisphere, region_config in structure_config.items():
                region_info = cls._get_region_info(
                    structure, hemisphere, region_config, shift
                )
                yield region_info

    @classmethod
    def _get_region_info(
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
        x_slice = cls._get_slice(x, shift)
        y_slice = cls._get_slice(y, shift)
        z_slice = cls._get_slice(z, shift)
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
    def _get_slice(center: int, shift: int) -> slice:
        return slice(center - shift, center + shift)


class WholeBrainDataFrame(pd.DataFrame):
    STIMULATIONS = ["Auditory", "Visual"]

    @classmethod
    def load_from_directory(
        cls, dirname: str, gen_random: bool = True, random_seed: int = 42
    ) -> pd.DataFrame:
        df = []
        for stimulation in cls.STIMULATIONS:
            stim_dirname = os.path.join(dirname, stimulation)
            subjects = os.listdir(stim_dirname)
            subjects = sorted(
                subjects, key=lambda x: int(re.findall(r"\d+$", x)[0])
            )
            subjects_data_path = cls._get_subjects_data_path(
                dirname=stim_dirname,
                stimulation=stimulation,
                subjects=subjects,
            )
            subjects_df = _batch_get_real_subject_data(
                paths_lst=subjects_data_path,
                stimulation=stimulation,
                subject_lst=subjects,
            )
            df.append(subjects_df)
            if gen_random:
                random_subjects_df = _get_random_subject_dataframe(
                    real_dataframe=subjects_df, random_seed=random_seed
                )
                df.append(random_subjects_df)
        return pd.concat(df, ignore_index=True)

    @classmethod
    def _get_subjects_data_path(
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


class BrainRegionsDataFrame(pd.DataFrame):
    @classmethod
    def load_from_directory(cls, dirname: str) -> pd.DataFrame:
        location_path = os.path.join(dirname, "location.yaml")
        location = LocationDataFrame.load_from_location_config(
            path=location_path
        )
        database = WholeBrainDataFrame.load_from_directory(dirname=dirname)

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


class PatternDataset(pd.DataFrame):
    @classmethod
    def load_from_dataframe(
        cls,
        df: pd.DataFrame,
        stimulation: _Stimulation,
        n_subject: int = 0,
        only_specific: bool = True,
    ) -> pd.DataFrame:
        df = df[df["data_type"] == "Real"]
        df = df[df["stimulation"] == stimulation]
        regions = df["region"].unique()
        if only_specific:
            match stimulation:
                case "Auditory":
                    specific_structure = "STG"
                case "Visual":
                    specific_structure = "FFA"
                case _:
                    raise ValueError(
                        f"Invalid stimulation type: {stimulation}"
                    )
            regions = [
                region
                for region in regions
                if get_structure(region) == specific_structure
            ]
        pattern_df_lst = []
        for region in regions:
            structure = get_structure(region)
            hemisphere = get_hemisphere(region)
            region_df = df[df["region"] == region]
            region_df_idx = region_df.index.to_list()

            if n_subject <= 0:
                region_df_combinations = combinations(
                    region_df_idx, len(region_df_idx) + n_subject
                )
            else:
                region_df_combinations = combinations(region_df_idx, n_subject)
            for pattern_subjects_idx in region_df_combinations:
                pattern_df = region_df.loc[list(pattern_subjects_idx)]
                pattern_data = np.nanmean(pattern_df["data"].tolist(), axis=0)
                pattern_data = (
                    pd.Series(
                        {
                            "region": region,
                            "structure": structure,
                            "hemisphere": hemisphere,
                            "data_type": "Pattern",
                            "stimulation": stimulation,
                            "subject": ",".join(
                                pattern_df["subject"].to_list()
                            ),
                            "data": pattern_data,
                        }
                    )
                    .to_frame()
                    .T
                )
                pattern_df_lst.append(pattern_data)
        return pd.concat(pattern_df_lst, ignore_index=True)


@dataclass
class IndexInfo:
    subject_idx: int
    data_type: Literal["Random", "Real", "Pattern"]
    stimulation: _Stimulation


def add_similarity_data(
    similarity_dataset: pd.DataFrame | None,
    matrix: np.ndarray,
    region: str,
    structure: str,
    hemisphere: str,
    modify_method: str,
    similarity: str,
    matrix_type: Literal["subject-subject", "pattern-subject"],
    axis_0_index_info: dict[int, IndexInfo],
    axis_1_index_info: dict[int, IndexInfo],
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
        axis_0_index_info: A dictionary of the similarity's index's
          information of the axis 0 of the matrix.
        axis_1_index_info: A dictionary of the similarity's index's
          information of the axis 1 of the matrix.

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
                "axis_0_indices": axis_0_index_info,
                "axis_1_indices": axis_1_index_info,
            }
        )
        .to_frame()
        .T
    )
    if similarity_dataset is not None:
        similarity_dataset = pd.concat(
            [similarity_dataset, data], ignore_index=True
        )
        return similarity_dataset
    else:
        return data
