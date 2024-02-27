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
