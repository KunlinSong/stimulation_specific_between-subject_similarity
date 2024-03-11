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

_DataType = Literal["Random", "Real"]
_Hemisphere = Literal["L", "R"]
_Region = Literal["FFA L", "FFA R", "STG L", "STG R"]
_Stimulation = Literal["Auditory", "Visual"]
_Structure = Literal["FFA", "STG"]
_PreprocessMethod = Literal["FFT", "Gradient", "Spatial Average"]


class _FFTKwargs(TypedDict):
    part: Literal["real", "imaginary", "both"]


class _GradientKwargs(TypedDict):
    axis: int | Literal["x", "y", "z"] | None
    keep_raw: bool


class _SpatialAverageKwargs(TypedDict):
    kernel_size: int
    sigma: float
