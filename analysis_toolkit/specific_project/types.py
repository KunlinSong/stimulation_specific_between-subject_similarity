import struct
from typing import (
    Any,
    Callable,
    Generator,
    Literal,
    NamedTuple,
    Optional,
    TypedDict,
    Union,
    overload,
)

from scipy.fftpack import shift


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


class _FFTKwargs(TypedDict):
    part: Literal["real", "imaginary", "both"]


class _GradientKwargs(TypedDict):
    axis: int | Literal["x", "y", "z"] | None
    keep_raw: bool


class _SpatialAverageKwargs(TypedDict):
    kernel_size: int
    sigma: float
