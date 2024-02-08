#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file contains the implementation of the _dataset module.
"""

__author__ = "Kunlin SONG"
__email__ = "kunlinsongcode@gmail.com"
__version__ = "1.0.0"
__license__ = "MIT"


import dataclasses
import os
import re
from typing import Callable, Literal, NamedTuple, Optional, TypedDict, overload

import nibabel as nib
import numpy as np
import pandas as pd
import yaml


class _LocationColumnsDict(TypedDict):
    name: str
    region: str
    structure: str
    hemisphere: str
    center_x: int
    center_y: int
    center_z: int
    shift: int
    slice_x: slice
    slice_y: slice
    slice_z: slice
    mask: np.ndarray


CenterConfig = NamedTuple(
    "CenterConfig",
    [("structure", str), ("hemisphere", str), ("hemisphere_config", dict)],
)


def _get_region(structure: str, hemisphere: str) -> str:
    return f"{structure} {hemisphere}"


class Location:
    """A class representing the location of a region (seed pattern) in
    the brain as a DataFrame from a YAML file.


    Attributes:
        FULL_SHAPE (tuple): The full shape of the brain.
    """

    FULL_SHAPE = (53, 63, 46)

    def __new__(cls, path: str) -> pd.DataFrame:
        """Create a new Location object, which is actually a DataFrame.

        Args:
            path (str): The path to the YAML file containing the location
            of the regions (seed patterns) in the brain.

        Returns:
            pd.DataFrame: A DataFrame containing the location of the
            regions (seed patterns) in the brain
        """
        with open(path, "r") as file:
            config: dict = yaml.safe_load(file)
        shift: int = config["Shift"]
        center_config: dict[str, dict] = config["Center"]
        location_df = pd.DataFrame(columns=_LocationColumnsDict.__annotations__.keys())
        for structure, hemisphere, hemisphere_config in cls._config_iterator(
            center_config
        ):
            region = _get_region(structure, hemisphere)
            center_x = hemisphere_config["x"]
            center_y = hemisphere_config["y"]
            center_z = hemisphere_config["z"]
            slice_x = cls._get_slice(center_x, shift)
            slice_y = cls._get_slice(center_y, shift)
            slice_z = cls._get_slice(center_z, shift)
            mask = cls._get_mask(slice_x, slice_y, slice_z)

            new_row: _LocationColumnsDict = {
                "name": region,
                "region": region,
                "structure": structure,
                "hemisphere": hemisphere,
                "center_x": center_x,
                "center_y": center_y,
                "center_z": center_z,
                "shift": shift,
                "slice_x": slice_x,
                "slice_y": slice_y,
                "slice_z": slice_z,
                "mask": mask,
            }
            new_row = pd.Series(new_row).to_frame().T
            location_df = pd.concat([location_df, new_row], ignore_index=True)
        location_df = location_df.set_index("name")
        return location_df

    @staticmethod
    def _config_iterator(config: dict[str, dict]) -> CenterConfig:
        """
        Iterate over the configuration dictionary and yield CenterConfig
        objects.

        Args:
            config (dict[str, dict]): The configuration dictionary
            containing information of structure, hemisphere, and the
            configuration of the hemisphere.

        Yields:
            CenterConfig: A CenterConfig object containing the structure,
            hemisphere, and the configuration of the hemisphere.

        """
        for structure, structure_config in config.items():
            for hemisphere, hemisphere_config in structure_config.items():
                yield CenterConfig(structure, hemisphere, hemisphere_config)

    @staticmethod
    def _get_slice(center: int, shift: int) -> slice:
        """Get a slice based on the center and shift.

        Args:
            center (int): The center of the slice, which is the center of
            the region (seed pattern) on the indicated axis.
            shift (int): The shift value, which is the half of the slice.

        Returns:
            slice: The slice object.
        """
        return slice(center - shift, center + shift)

    @staticmethod
    def _get_mask(slice_x: slice, slice_y: slice, slice_z: slice) -> np.ndarray:
        """Get a mask based on the slices.

        Args:
            slice_x (slice): The slice for the x-axis.
            slice_y (slice): The slice for the y-axis.
            slice_z (slice): The slice for the z-axis.

        Returns:
            np.ndarray: The mask array with the same shape as the brain,
            where the values of the region (seed pattern) are True and
            the rest are False.
        """
        mask = np.full(Location.FULL_SHAPE, False, dtype=bool)
        mask[slice_x, slice_y, slice_z] = True
        return mask


class _DatabaseColumnsDict(TypedDict):
    name: str
    type: Literal["Random", "Real"]
    stimulation: str
    subject: str
    data: np.ndarray


Data = NamedTuple("Data", [("Random", np.ndarray), ("Real", np.ndarray)])


class _Database:
    """
    A class representing a database of data.

    Args:
        dirname (str): The directory name.
        stimulation (str): The stimulation type.
        get_path_func (Callable): A function to get the path for each subject.

    Attributes:
        RANDOM_SEED (int): The random seed for the random number generator
        used to shuffle the real data to create the random data.

    Returns:
        pd.DataFrame: The database as a pandas DataFrame.
    """

    RANDOM_SEED = 42

    def __new__(
        cls, dirname: str, stimulation: str, get_path_func: Callable
    ) -> pd.DataFrame:
        visual_df = pd.DataFrame(columns=_DatabaseColumnsDict.__annotations__.keys())
        for subject in os.listdir(dirname):
            path = get_path_func(subject)
            data = cls._get_data(path)
            for data_type in data._fields:
                new_row: _DatabaseColumnsDict = {
                    "name": subject,
                    "type": data_type,
                    "stimulation": stimulation,
                    "subject": subject,
                    "data": getattr(data, data_type),
                }
                new_row = pd.Series(new_row).to_frame().T
                visual_df = pd.concat([visual_df, new_row], ignore_index=True)
        visual_df = visual_df.set_index("name")
        return visual_df

    def _get_data(self, path: str) -> Data:
        """
        Get data from a given path.

        Args:
            path (str): The path to the data.

        Returns:
            Data: The data object.
        """
        data = nib.load(path)
        data_array_real: np.ndarray = data.get_fdata(dtype=np.float32)

        data_array_random = data_array_real.copy()
        not_nan_indices = np.argwhere(~np.isnan(data_array_random))
        random_indices = not_nan_indices.copy()
        random_generator = np.random.default_rng(self.RANDOM_SEED)
        random_generator.shuffle(random_indices)
        data_array_random[not_nan_indices] = data_array_random[random_indices]
        return Data(Random=data_array_random, Real=data_array_real)


class _VisualDatabase(_Database):
    """
    A class representing a database whose subjects were got
    visual-specific stimulation.

    This class inherits from the _Database class.

    Attributes:
        FILENAME (str): The name of the file NIfTI file containing the
        data.
        STIMULATION (str): The type of stimulation for the database.
    """

    FILENAME = "con_0006.img"
    STIMULATION = "Visual"

    def __new__(cls, dirname: str) -> pd.DataFrame:
        """
        Create a new instance of the _VisualDatabase class.

        Args:
            dirname (str): The directory name where the data is in.

        Returns:
            pd.DataFrame: The database as a pandas DataFrame.
        """
        get_path_func = lambda subject: cls._get_path(dirname=dirname, subject=subject)
        return super().__new__(
            cls,
            dirname=dirname,
            stimulation=cls.STIMULATION,
            get_path_func=get_path_func,
        )

    def _get_path(self, dirname: str, subject: str) -> str:
        """
        Get the path of the file for a specific subject.

        Args:
            dirname (str): The directory name where the subjects' data
            are in.
            subject (str): The subject for which to get the file path.

        Returns:
            str: The path of the file for the specified subject.
        """
        return os.path.join(dirname, subject, self.FILENAME)


class _AuditoryDatabase(_Database):
    """
    A class representing a database whose subjects were got
    auditory-specific stimulation.

    This class inherits from the _Database class.

    Attributes:
        SUBJECT_PATTERN (str): The pattern for the subject name.
        STIMULATION (str): The type of stimulation for the database.
    """

    SUBJECT_PATTERN = r"Subject_(\d+)"
    STIMULATION = "Auditory"

    def __new__(cls, dirname: str) -> pd.DataFrame:
        """
        Create a new instance of the _AuditoryDatabase class.

        Args:
            dirname (str): The directory name where the data is in.

        Returns:
            pd.DataFrame: The database as a pandas DataFrame.
        """
        get_path_func = lambda subject: cls._get_path(dirname=dirname, subject=subject)
        return super().__new__(
            cls,
            dirname=dirname,
            stimulation=cls.STIMULATION,
            get_path_func=get_path_func,
        )

    def _get_path(self, dirname: str, subject: str) -> str:
        """
        Get the path of the file for a specific subject.

        Args:
            dirname (str): The directory name where the subjects' data
            are in.
            subject (str): The subject for which to get the file path.

        Returns:
            str: The path of the file for the specified subject.
        """
        subject_num = re.findall(self.SUBJECT_PATTERN, subject)[0]
        subject_num = int(subject_num)
        filename = f"Words_{subject_num}.nii"
        return os.path.join(dirname, subject, filename)


Slices = NamedTuple("Slices", [("x", slice), ("y", slice), ("z", slice)])


class Dataset:
    """
    A class representing a dataset.

    Attributes:
        LOCATION_CONFIG_FILENAME (str): The filename of the location
        configuration file.
    """

    LOCATION_CONFIG_FILENAME = "location.yaml"

    def __init__(self, dirname: str) -> None:
        """
        Initializes the Dataset object.

        Args:
            dirname (str): The directory name of the dataset.
        """
        self._location_config = Location(
            os.path.join(dirname, self.LOCATION_CONFIG_FILENAME)
        )
        visual_db = _VisualDatabase(os.path.join(dirname, "Visual"))
        auditory_db = _AuditoryDatabase(os.path.join(dirname, "Auditory"))
        self._database = pd.concat([visual_db, auditory_db])

    @property
    def location_config(self) -> pd.DataFrame:
        """
        Returns the location configuration.

        Returns:
            pd.DataFrame: The location configuration.
        """
        return self._location_config

    @property
    def database(self) -> pd.DataFrame:
        """
        Returns the database.

        Returns:
            pd.DataFrame: The database.
        """
        return self._database

    @property
    def regions(self) -> list[str]:
        """
        Returns a sorted list of regions.

        Returns:
            list[str]: A sorted list of regions.
        """
        regions = self._location_config.index.tolist()
        return sorted(regions)

    def _get_region_mask(self, region: str) -> np.ndarray:
        """
        Returns the mask for a specific region.

        Args:
            region (str): The region name.

        Returns:
            np.ndarray: The mask for the region.
        """
        return self._location_config.loc[region, "mask"].copy()

    def _get_region_slices(self, region: str) -> Slices:
        """
        Returns the slices for a specific region.

        Args:
            region (str): The region name.

        Returns:
            Slices: The slices for the region.
        """
        return Slices(
            x=self._location_config.loc[region, "slice_x"],
            y=self._location_config.loc[region, "slice_y"],
            z=self._location_config.loc[region, "slice_z"],
        )

    @overload
    def _get_region(self, region: str) -> str: ...

    @overload
    def _get_region(self, structure: str, hemisphere: str) -> str: ...

    def _get_region(
        self,
        region: Optional[str] = None,
        structure: Optional[str] = None,
        hemisphere: Optional[str] = None,
    ) -> str:
        """
        Returns the region based on the given parameters. If the region
        is given, it will be returned. Otherwise, the structure and
        hemisphere will be used to get the region.

        Args:
            region (str, optional): The region name. Defaults to None.
            structure (str, optional): The structure name. Defaults to
            None.
            hemisphere (str, optional): The hemisphere name. Defaults to
            None.

        Returns:
            str: The region name.
        """
        if region is not None:
            return region
        else:
            return _get_region(structure, hemisphere)

    @overload
    def get_region_mask(self, region: str) -> np.ndarray: ...

    @overload
    def get_region_mask(self, structure: str, hemisphere: str) -> np.ndarray: ...

    def get_region_mask(
        self,
        region: Optional[str] = None,
        structure: Optional[str] = None,
        hemisphere: Optional[str] = None,
    ) -> np.ndarray:
        """
        Returns the mask for a specific region based on the given
        parameters.

        Args:
            region (str, optional): The region name. Defaults to None.
            structure (str, optional): The structure name. Defaults to
            None.
            hemisphere (str, optional): The hemisphere name. Defaults to
            None.

        Returns:
            np.ndarray: The mask for the region.
        """
        region = self._get_region(
            region=region, structure=structure, hemisphere=hemisphere
        )
        return self._get_region_mask(region)

    @overload
    def get_region_slices(self, region: str) -> Slices: ...

    @overload
    def get_region_slices(self, structure: str, hemisphere: str) -> Slices: ...

    def get_region_slices(
        self,
        region: Optional[str] = None,
        structure: Optional[str] = None,
        hemisphere: Optional[str] = None,
    ) -> Slices:
        """
        Returns the slices for a specific region based on the given
        parameters.

        Args:
            region (str, optional): The region name. Defaults to None.
            structure (str, optional): The structure name. Defaults to
            None.
            hemisphere (str, optional): The hemisphere name. Defaults to
            None.

        Returns:
            Slices: The slices for the region.
        """
        region = self._get_region(
            region=region, structure=structure, hemisphere=hemisphere
        )
        return self._get_region_slices(region)

    @property
    def structures(self) -> list[str]:
        """
        Returns a sorted list of structures.

        Returns:
            list[str]: A sorted list of structures.
        """
        structures = self._location_config["structure"].unique().tolist()
        return sorted(structures)

    @property
    def hemispheres(self) -> list[str]:
        """
        Returns a sorted list of hemispheres.

        Returns:
            list[str]: A sorted list of hemispheres.
        """
        hemispheres = self._location_config["hemisphere"].unique().tolist()
        return sorted(hemispheres)

    @property
    def data_types(self) -> list[str]:
        """
        Returns a sorted list of data types.

        Returns:
            list[str]: A sorted list of data types.
        """
        data_types = self._database["type"].unique().tolist()
        return sorted(data_types)

    @property
    def stimulations(self) -> list[str]:
        """
        Returns a sorted list of stimulations.

        Returns:
            list[str]: A sorted list of stimulations.
        """
        stimulations = self._database["stimulation"].unique().tolist()
        return sorted(stimulations)

    def subjects(self, stimulation: str) -> list[str]:
        """
        Returns a sorted list of subjects for a specific stimulation.

        Args:
            stimulation (str): The stimulation name.

        Returns:
            list[str]: A sorted list of subjects.
        """
        subjects = (
            self._database[(self._database["stimulation"] == stimulation)]["subject"]
            .unique()
            .tolist()
        )
        return sorted(subjects)

    def subject_names(self, data_type: str, stimulation: str) -> list[str]:
        """
        Returns a sorted list of subject names for a specific data type
        and stimulation.

        Args:
            data_type (str): The data type.
            stimulation (str): The stimulation name.

        Returns:
            list[str]: A sorted list of subject names.
        """
        subject_names = self._database[
            (self._database["type"] == data_type)
            & (self._database["stimulation"] == stimulation)
        ].index.tolist()
        return sorted(subject_names)

    def get_subject_name(self, data_type: str, stimulation: str, subject: str) -> str:
        """
        Returns the subject name based on the given data type,
        stimulation, and subject.

        Args:
            data_type (str): The data type.
            stimulation (str): The stimulation name.
            subject (str): The subject name.

        Returns:
            str: The subject name.
        """
        return self._database[
            (self._database["type"] == data_type)
            & (self._database["stimulation"] == stimulation)
            & (self._database["subject"] == subject)
        ].index[0]

    @overload
    def get_whole_brain_data(
        self, data_type: str, stimulation: str, subject: str
    ) -> np.ndarray: ...

    @overload
    def get_whole_brain_data(self, subject_name: str) -> np.ndarray: ...

    def get_whole_brain_data(
        self,
        subject_name: Optional[str] = None,
        data_type: Optional[str] = None,
        stimulation: Optional[str] = None,
        subject: Optional[str] = None,
    ) -> np.ndarray:
        """
        Returns the whole brain data for a specific subject or based on
        the given parameters.

        Args:
            subject_name (str, optional): The subject name. Defaults to
            None.
            data_type (str, optional): The data type. Defaults to None.
            stimulation (str, optional): The stimulation name. Defaults
            to None.
            subject (str, optional): The subject name. Defaults to None.

        Returns:
            np.ndarray: The whole brain data.
        """
        if subject_name is None:
            subject_name = self.get_subject_name(
                data_type=data_type, stimulation=stimulation, subject=subject
            )
        return self._database.loc[subject_name, "data"].copy()

    @overload
    def get_region_data(
        self,
        subject_name: str,
        region: str,
    ) -> np.ndarray: ...

    @overload
    def get_region_data(
        self,
        subject_name: str,
        structure: str,
        hemisphere: str,
    ) -> np.ndarray: ...

    @overload
    def get_region_data(
        self,
        data_type: str,
        stimulation: str,
        subject: str,
        region: str,
    ) -> np.ndarray: ...

    @overload
    def get_region_data(
        self,
        data_type: str,
        stimulation: str,
        subject: str,
        structure: str,
        hemisphere: str,
    ) -> np.ndarray: ...

    def get_region_data(
        self,
        subject_name: Optional[str] = None,
        data_type: Optional[str] = None,
        stimulation: Optional[str] = None,
        subject: Optional[str] = None,
        region: Optional[str] = None,
        structure: Optional[str] = None,
        hemisphere: Optional[str] = None,
    ) -> np.ndarray:
        """
        Returns the region data for a specific subject or based on the
        given parameters.

        Args:
            subject_name (str, optional): The subject name. Defaults to
            None.
            data_type (str, optional): The data type. Defaults to None.
            stimulation (str, optional): The stimulation name. Defaults
            to None.
            subject (str, optional): The subject name. Defaults to None.
            region (str, optional): The region name. Defaults to None.
            structure (str, optional): The structure name. Defaults to
            None.
            hemisphere (str, optional): The hemisphere name. Defaults to
            None.

        Returns:
            np.ndarray: The region data.
        """
        region = self._get_region(
            region=region, structure=structure, hemisphere=hemisphere
        )
        region_slices = self._get_region_slices(region=region)
        whole_brain_data: np.ndarray = self.get_whole_brain_data(
            subject_name=subject_name,
            data_type=data_type,
            stimulation=stimulation,
            subject=subject,
        )
        region_data = whole_brain_data[
            region_slices.x, region_slices.y, region_slices.z
        ]
        return region_data.copy()


@dataclasses.dataclass
class SubjectSimilarityVector:
    subject_name: str
    structure: str
    hemisphere: str
    subject: str
    idx: int
    similarity_vector: np.ndarray


@dataclasses.dataclass
class SimilarityMatrix:
    matrix: np.ndarray
    region: str
    structure: str
    hemisphere: str
    stimulation_slices: dict[str, slice]
    subject_vectors: dict[int, SubjectSimilarityVector]
