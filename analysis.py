# Analysis
# %%

# Basic Parameters
DATA_DIRNAME = "./Dataset"
GEN_DIRNAME = "./Generates"
HEATMAP_COLOR_MAP = "seismic"
GROUP_COLORS = {
    "Real Visual": "#E26844",
    "Real Auditory": "#329845",
    "Random Visual": "#FCDC89",
    "Random Auditory": "#AED185",
}

import os

# %%
# Importing Libraries
from dataclasses import dataclass
from functools import partial
from itertools import combinations, combinations_with_replacement, product
from tkinter.scrolledtext import ScrolledText

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots
import seaborn as sns
import shap
import sklearn
import sklearn.metrics
import sklearn.tree
from matplotlib import ticker as mticker
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from statannotations.Annotator import Annotator

import analysis_toolkit

# %%
# Load the data
dataset = analysis_toolkit.dataset.BrainRegionsDataFrame.load_from_directory(
    dirname=DATA_DIRNAME
)
# %%
# Create "Generates" directory if it does not exist
os.makedirs(GEN_DIRNAME, exist_ok=True)

# %%
# Use "science" style for plots
plt.style.use("science")


# %%
# Statistical properties of the dataset
def save_statistical_properties(df: pd.DataFrame, save_dirname: str) -> None:
    statistical_properties_df = pd.DataFrame(
        columns=[
            "Region",
            "DataType",
            "Stimulation",
            "NumberOfSubjects",
            "Mean",
            "Std",
            "0%",
            "25%",
            "50%",
            "75%",
            "100%",
        ]
    )
    format_info = lambda x: f"{x:.2f}"
    for region in df["region"].unique():
        region_df = df[df["region"] == region]
        for data_type in df["data_type"].unique():
            data_type_df = region_df[region_df["data_type"] == data_type]
            for stimulation in df["stimulation"].unique():
                stimulation_df = data_type_df[
                    data_type_df["stimulation"] == stimulation
                ]
                data = stimulation_df["data"].values
                data = np.stack(data, axis=0)
                new_row = pd.Series(
                    {
                        "Region": region,
                        "DataType": data_type,
                        "Stimulation": stimulation,
                        "NumberOfSubjects": len(data),
                        "Mean": format_info(np.nanmean(data)),
                        "Std": format_info(np.nanstd(data)),
                        "0%": format_info(np.nanmin(data)),
                        "25%": format_info(np.nanpercentile(data, 25)),
                        "50%": format_info(np.nanpercentile(data, 50)),
                        "75%": format_info(np.nanpercentile(data, 75)),
                        "100%": format_info(np.nanmax(data)),
                    }
                )
                statistical_properties_df = pd.concat(
                    [statistical_properties_df, new_row.to_frame().T],
                    ignore_index=True,
                )
    statistical_properties_df.to_csv(
        os.path.join(save_dirname, "basic_info.csv"), index=False
    )


save_statistical_properties(df=dataset, save_dirname=GEN_DIRNAME)
del save_statistical_properties


# %%
# Get Pattern
def get_pattern_df(df: pd.DataFrame):
    N_SUBJECT_LST = [0, -1]
    pattern_df_lst = []
    for stimulation in df["stimulation"].unique():
        for n_subject in N_SUBJECT_LST:
            pattern_df = (
                analysis_toolkit.dataset.PatternDataset.load_from_dataframe(
                    df=df,
                    stimulation=stimulation,
                    n_subject=n_subject,
                    only_specific=False,
                )
            )
            pattern_df_lst.append(pattern_df)
    return pd.concat(pattern_df_lst, ignore_index=True)


pattern_df = get_pattern_df(dataset)
del get_pattern_df
dataset = pd.concat([dataset, pattern_df], ignore_index=True)
del pattern_df


# %%
# Get subject-subject Similarity
def get_similarity_df(df: pd.DataFrame):
    SIMILARITY_METHODS = ["cs", "pcc"]
    PREPROCESS_METHODS = [None, "FFT", "Gradient", "Spatial Average"]
    PREPROCESS_KWARGS = {
        "FFT": {"part": "both"},
        "Gradient": {"axis": None, "keep_raw": True},
        "Spatial Average": {"kernel_size": 3, "sigma": 1.0, "keep_raw": True},
    }
    df = df.copy()
    df = df[df["data_type"] != "Pattern"]
    similarity_df = None
    for region in df["region"].unique():
        region_df: pd.DataFrame = df[df["region"] == region].copy()
        region_df.sort_values(
            by=["data_type", "stimulation", "subject"], inplace=True
        )
        for similarity_method in SIMILARITY_METHODS:
            similarity_func = getattr(
                analysis_toolkit.similarity, similarity_method
            )
            for preprocess_method in PREPROCESS_METHODS:
                if preprocess_method is None:
                    func = partial(similarity_func, preprocess_method=None)
                else:
                    func = partial(
                        similarity_func,
                        preprocess_method=preprocess_method,
                        preprocess_kwargs=PREPROCESS_KWARGS[preprocess_method],
                    )
                similarity_mat = np.empty((len(region_df), len(region_df)))
                axis_0_index_info = {}
                axis_1_index_info = {}
                for (mat_idx_0, (subject_idx_0, row_0)), (
                    mat_idx_1,
                    (subject_idx_1, row_1),
                ) in combinations_with_replacement(
                    enumerate(region_df.iterrows()), 2
                ):
                    subject_0 = row_0["data"]
                    subject_1 = row_1["data"]
                    similarity_value = func(subject_0, subject_1)
                    similarity_mat[mat_idx_0, mat_idx_1] = similarity_value
                    similarity_mat[mat_idx_1, mat_idx_0] = similarity_value
                    axis_0_index_info[mat_idx_0] = (
                        analysis_toolkit.dataset.IndexInfo(
                            subject_idx=subject_idx_0,
                            data_type=row_0["data_type"],
                            stimulation=row_0["stimulation"],
                        )
                    )
                    axis_1_index_info[mat_idx_1] = (
                        analysis_toolkit.dataset.IndexInfo(
                            subject_idx=subject_idx_1,
                            data_type=row_1["data_type"],
                            stimulation=row_1["stimulation"],
                        )
                    )
                similarity_df = analysis_toolkit.dataset.add_similarity_data(
                    similarity_dataset=similarity_df,
                    matrix=similarity_mat,
                    region=region,
                    structure=analysis_toolkit.dataset.get_structure(
                        region=region
                    ),
                    hemisphere=analysis_toolkit.dataset.get_hemisphere(
                        region=region
                    ),
                    modify_method=(
                        preprocess_method
                        if preprocess_method is not None
                        else "None"
                    ),
                    similarity=similarity_method,
                    matrix_type="subject-subject",
                    axis_0_index_info=axis_0_index_info,
                    axis_1_index_info=axis_1_index_info,
                )
    return similarity_df


similarity_df = get_similarity_df(dataset)
del get_similarity_df


# %%
# Get subject-pattern Similarity
def get_pattern_similarity_df(df: pd.DataFrame):
    SIMILARITY_METHODS = ["cs", "pcc"]
    PREPROCESS_METHODS = [None, "FFT", "Gradient", "Spatial Average"]
    PREPROCESS_KWARGS = {
        "FFT": {"part": "both"},
        "Gradient": {"axis": None, "keep_raw": True},
        "Spatial Average": {"kernel_size": 3, "sigma": 1.0, "keep_raw": True},
    }
    df = df.copy()
    similarity_df = None
    for region in df["region"].unique():
        region_df: pd.DataFrame = df[df["region"] == region].copy()
        pattern_df = region_df[region_df["data_type"] == "Pattern"].copy()
        real_df = region_df[region_df["data_type"] != "Pattern"].copy()
        pattern_df.sort_values(by=["stimulation", "subject"], inplace=True)
        real_df.sort_values(by=["stimulation", "subject"], inplace=True)
        del region_df
        for similarity_method in SIMILARITY_METHODS:
            similarity_func = getattr(
                analysis_toolkit.similarity, similarity_method
            )
            for preprocess_method in PREPROCESS_METHODS:
                if preprocess_method is None:
                    func = partial(similarity_func, preprocess_method=None)
                else:
                    func = partial(
                        similarity_func,
                        preprocess_method=preprocess_method,
                        preprocess_kwargs=PREPROCESS_KWARGS[preprocess_method],
                    )
                similarity_mat = np.empty((len(pattern_df), len(real_df)))
                axis_0_index_info = {}
                axis_1_index_info = {}
                for (mat_idx_0, (subject_idx_0, row_0)), (
                    mat_idx_1,
                    (subject_idx_1, row_1),
                ) in product(
                    enumerate(pattern_df.iterrows()),
                    enumerate(real_df.iterrows()),
                ):
                    subject_0 = row_0["data"]
                    subject_1 = row_1["data"]
                    similarity_value = func(subject_0, subject_1)
                    similarity_mat[mat_idx_0, mat_idx_1] = similarity_value
                    axis_0_index_info[mat_idx_0] = (
                        analysis_toolkit.dataset.IndexInfo(
                            subject_idx=subject_idx_0,
                            data_type=row_0["data_type"],
                            stimulation=row_0["stimulation"],
                        )
                    )
                    axis_1_index_info[mat_idx_1] = (
                        analysis_toolkit.dataset.IndexInfo(
                            subject_idx=subject_idx_1,
                            data_type=row_1["data_type"],
                            stimulation=row_1["stimulation"],
                        )
                    )
                similarity_df = analysis_toolkit.dataset.add_similarity_data(
                    similarity_dataset=similarity_df,
                    matrix=similarity_mat,
                    region=region,
                    structure=analysis_toolkit.dataset.get_structure(
                        region=region
                    ),
                    hemisphere=analysis_toolkit.dataset.get_hemisphere(
                        region=region
                    ),
                    modify_method=(
                        preprocess_method
                        if preprocess_method is not None
                        else "None"
                    ),
                    similarity=similarity_method,
                    matrix_type="pattern-subject",
                    axis_0_index_info=axis_0_index_info,
                    axis_1_index_info=axis_1_index_info,
                )
    return similarity_df


pattern_similarity_df = get_pattern_similarity_df(dataset)
del get_pattern_similarity_df
similarity_df = pd.concat(
    [similarity_df, pattern_similarity_df], ignore_index=True
)
del pattern_similarity_df

# %%
# Plot similarity matrix


def verify_datatype_and_stimulation(
    item,
    data_type: analysis_toolkit.dataset._dataset._DataType,
    stimulation: analysis_toolkit.dataset._dataset._Stimulation,
) -> bool:
    idx_info = item[1]
    return (getattr(idx_info, "data_type") == data_type) and (
        getattr(idx_info, "stimulation") == stimulation
    )


# %%


def plot_subject_subject_heatmap_and_save(
    similarity_df: pd.DataFrame,
    save_dirname: str,
    label_width: int = 2,
    figsize: tuple[int, int] | None = None,
    split_width: float = 1,
    axis_width: float = 2,
) -> None:
    similarity_df = similarity_df[
        similarity_df["matrix_type"] == "subject-subject"
    ].copy()
    for idx, row in similarity_df.iterrows():
        with open(os.path.join(save_dirname, f"{idx:02d}.txt"), "w") as f:
            f.write(f"Region: {row['region']}\n")
            f.write(f"Structure: {row['structure']}\n")
            f.write(f"Hemisphere: {row['hemisphere']}\n")
            f.write(f"Similarity: {row['similarity']}\n")
            f.write(f"Preprocess: {row['modify_method']}\n")
            f.write(f"Matrix Type: {row['matrix_type']}\n")

        fig, ax = plt.subplots(figsize=figsize)
        mat = row["matrix"]
        padded_mat = np.pad(
            array=mat,
            pad_width=((0, label_width), (label_width, 0)),
            mode="constant",
            constant_values=np.nan,
        )

        data_type_lst = list(
            set(
                v.data_type
                for v in row["axis_0_indices"].values()
                if v.data_type != "Pattern"
            )
        )
        stimulation_lst = list(
            set(v.stimulation for v in row["axis_0_indices"].values())
        )

        for data_type in data_type_lst:
            for stimulation in stimulation_lst:
                indices_info = dict(
                    filter(
                        partial(
                            verify_datatype_and_stimulation,
                            data_type=data_type,
                            stimulation=stimulation,
                        ),
                        row["axis_0_indices"].items(),
                    )
                )
                start = min(indices_info.keys())
                end = max(indices_info.keys()) + 1

                label_mat = np.full_like(padded_mat, fill_value=np.nan)
                label_mat[start:end, 0:label_width] = 1
                sns.heatmap(
                    label_mat,
                    ax=ax,
                    cmap=[GROUP_COLORS[f"{data_type} {stimulation}"]],
                    square=True,
                    cbar=False,
                    mask=np.nonzero(padded_mat) is False,
                    zorder=0,
                )
                ax.axhline(
                    start,
                    color="black",
                    linewidth=split_width,
                    zorder=2,
                )
                ax.axhline(
                    end,
                    color="black",
                    linewidth=split_width,
                    zorder=2,
                )

        data_type_lst = list(
            set(v.data_type for v in row["axis_1_indices"].values())
        )
        stimulation_lst = list(
            set(v.stimulation for v in row["axis_1_indices"].values())
        )

        for data_type in data_type_lst:
            for stimulation in stimulation_lst:
                indices_info = dict(
                    filter(
                        partial(
                            verify_datatype_and_stimulation,
                            data_type=data_type,
                            stimulation=stimulation,
                        ),
                        row["axis_1_indices"].items(),
                    )
                )
                start = min(indices_info.keys())
                end = max(indices_info.keys()) + 1

                label_mat = np.full_like(padded_mat, fill_value=np.nan)
                label_mat[
                    -label_width:, start + label_width : end + label_width
                ] = 1
                sns.heatmap(
                    label_mat,
                    ax=ax,
                    cmap=[GROUP_COLORS[f"{data_type} {stimulation}"]],
                    square=True,
                    cbar=False,
                    mask=np.nonzero(padded_mat) is False,
                    zorder=0,
                )
                ax.axvline(
                    start + label_width,
                    color="black",
                    linewidth=split_width,
                    zorder=2,
                )
                ax.axvline(
                    end + label_width,
                    color="black",
                    linewidth=split_width,
                    zorder=2,
                )

        sns.heatmap(
            padded_mat,
            ax=ax,
            square=True,
            cbar=False,
            vmax=1,
            vmin=-1,
            cmap=HEATMAP_COLOR_MAP,
            mask=np.nonzero(padded_mat) is True,
            zorder=1,
        )

        ax.axvline(label_width, color="black", linewidth=axis_width, zorder=2)
        ax.axhline(mat.shape[0], color="black", linewidth=axis_width, zorder=2)
        ax.vlines(
            0, 0, mat.shape[0], color="black", linewidth=axis_width, zorder=2
        )
        ax.hlines(
            mat.shape[0] + label_width,
            label_width,
            mat.shape[1] + label_width,
            color="black",
            linewidth=axis_width,
            zorder=2,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        fig.savefig(os.path.join(save_dirname, f"{idx:02d} heatmap.svg"))


plot_subject_subject_heatmap_and_save(
    similarity_df=similarity_df, save_dirname=GEN_DIRNAME, figsize=(10, 10)
)
del plot_subject_subject_heatmap_and_save
# %%
# Plot pattern-subject similarity matrix


def plot_pattern_subject_heatmap_and_save(
    df: pd.DataFrame,
    similarity_df: pd.DataFrame,
    save_dirname: str,
    label_width: int = 1,
    figsize: tuple[int, int] | None = None,
    split_width: float = 1,
    axis_width: float = 2,
) -> None:
    similarity_df = similarity_df[
        similarity_df["matrix_type"] == "pattern-subject"
    ].copy()
    df = df.copy()
    for region in similarity_df["region"].unique():
        structure = analysis_toolkit.dataset.get_structure(region)
        match structure:
            case "FFA":
                pattern_stimulation = "Visual"
            case "STG":
                pattern_stimulation = "Auditory"
        del structure
        real_pattern_stimulation_df = df[
            (df["data_type"] == "Real")
            & (df["stimulation"] == pattern_stimulation)
        ]
        n_subjects = real_pattern_stimulation_df["subject"].nunique()
        del real_pattern_stimulation_df
        region_similarity_df = similarity_df[similarity_df["region"] == region]
        pattern_subject_similarity_df = pd.DataFrame(
            columns=[
                "subjects_similarity",
                "similarity",
                "preprocess_method",
                "data_type",
                "stimulation",
            ]
        )
        for similarity_method in similarity_df["similarity"].unique():
            for preprocess_method in similarity_df["modify_method"].unique():
                method_similarity_df = region_similarity_df[
                    (region_similarity_df["similarity"] == similarity_method)
                    & (
                        region_similarity_df["modify_method"]
                        == preprocess_method
                    )
                ]
                axis_0_index_info = method_similarity_df[
                    "axis_0_indices"
                ].iloc[0]
                axis_1_index_info = method_similarity_df[
                    "axis_1_indices"
                ].iloc[0]
                method_similarity_mat = method_similarity_df["matrix"].iloc[0]
                axis_0_stimulation_index_info = dict(
                    filter(
                        partial(
                            verify_datatype_and_stimulation,
                            data_type="Pattern",
                            stimulation=pattern_stimulation,
                        ),
                        axis_0_index_info.items(),
                    )
                )
                axis_0_df = df.loc[
                    [
                        idx_info.subject_idx
                        for idx_info in axis_0_stimulation_index_info.values()
                    ]
                ]
                for data_type in ["Random", "Real"]:
                    for stimulation in df["stimulation"].unique():
                        subjects_similarity = []
                        axis_1_stimulation_index_info = dict(
                            filter(
                                partial(
                                    verify_datatype_and_stimulation,
                                    data_type=data_type,
                                    stimulation=stimulation,
                                ),
                                axis_1_index_info.items(),
                            )
                        )
                        for (
                            mat_idx_1,
                            idx_info_1,
                        ) in axis_1_stimulation_index_info.items():
                            df_idx_1 = idx_info_1.subject_idx
                            subject_name_1 = df.loc[df_idx_1, "subject"]
                            del df_idx_1
                            for df_idx_0, row_0 in axis_0_df.iterrows():
                                axis_0_subject_lst = row_0["subject"].split(
                                    ","
                                )
                                pattern_n_subjects = len(axis_0_subject_lst)
                                try:
                                    if (data_type == "Real") & (
                                        stimulation == pattern_stimulation
                                    ):
                                        if (
                                            subject_name_1
                                            in axis_0_subject_lst
                                        ):
                                            raise ValueError
                                        else:
                                            needed_n_subjects = n_subjects - 1
                                    else:
                                        needed_n_subjects = n_subjects
                                    if pattern_n_subjects != needed_n_subjects:
                                        raise ValueError
                                except ValueError:
                                    continue

                                for (
                                    mat_idx_0,
                                    idx_info_0,
                                ) in axis_0_stimulation_index_info.items():
                                    if idx_info_0.subject_idx == df_idx_0:
                                        subjects_similarity.append(
                                            method_similarity_mat[
                                                mat_idx_0, mat_idx_1
                                            ]
                                        )
                                        break
                        new_row = (
                            pd.Series(
                                {
                                    "subjects_similarity": np.array(
                                        subjects_similarity
                                    ),
                                    "similarity": similarity_method,
                                    "preprocess_method": preprocess_method,
                                    "data_type": data_type,
                                    "stimulation": stimulation,
                                }
                            )
                            .to_frame()
                            .T
                        )
                        pattern_subject_similarity_df = pd.concat(
                            [pattern_subject_similarity_df, new_row],
                            ignore_index=True,
                        )
        fig, ax = plt.subplots(figsize=figsize)
        similarity_mat = []
        method_lst = []
        for similarity_method in pattern_subject_similarity_df[
            "similarity"
        ].unique():
            for preprocess_method in pattern_subject_similarity_df[
                "preprocess_method"
            ].unique():
                method_lst.append(
                    similarity_method
                    if preprocess_method == "None"
                    else f"{similarity_method} ({preprocess_method})"
                )
                method_vector = []
                pos = 0
                label_pos_dict = {}
                for data_type in pattern_subject_similarity_df[
                    "data_type"
                ].unique():
                    for stimulation in pattern_subject_similarity_df[
                        "stimulation"
                    ].unique():
                        similarity_value = pattern_subject_similarity_df[
                            (
                                pattern_subject_similarity_df["data_type"]
                                == data_type
                            )
                            & (
                                pattern_subject_similarity_df["similarity"]
                                == similarity_method
                            )
                            & (
                                pattern_subject_similarity_df[
                                    "preprocess_method"
                                ]
                                == preprocess_method
                            )
                            & (
                                pattern_subject_similarity_df["stimulation"]
                                == stimulation
                            )
                        ]["subjects_similarity"].iloc[0]
                        method_vector.append(similarity_value)
                        label_pos_dict[f"{data_type} {stimulation}"] = [
                            pos,
                            pos + len(similarity_value),
                        ]
                        pos += len(similarity_value)
                method_vector = np.concatenate(method_vector, axis=0)
                similarity_mat.append(method_vector)
        similarity_mat = np.stack(similarity_mat, axis=0)
        padded_mat = np.pad(
            similarity_mat,
            ((0, label_width), (0, 0)),
            mode="constant",
            constant_values=np.nan,
        )
        for data_type in pattern_subject_similarity_df["data_type"].unique():
            for stimulation in pattern_subject_similarity_df[
                "stimulation"
            ].unique():
                label = f"{data_type} {stimulation}"
                label_mat = np.full_like(padded_mat, fill_value=np.nan)
                label_mat[
                    -label_width:,
                    label_pos_dict[label][0] : label_pos_dict[label][1],
                ] = 1
                sns.heatmap(
                    label_mat,
                    ax=ax,
                    cmap=[GROUP_COLORS[label]],
                    square=True,
                    cbar=False,
                    mask=np.nonzero(padded_mat) is False,
                    zorder=0,
                )
                ax.axvline(
                    label_pos_dict[label][0],
                    color="black",
                    linewidth=split_width,
                    zorder=2,
                )
                ax.axvline(
                    label_pos_dict[label][1],
                    color="black",
                    linewidth=split_width,
                    zorder=2,
                )
        sns.heatmap(
            padded_mat,
            ax=ax,
            square=True,
            cbar=False,
            vmax=1,
            vmin=-1,
            cmap=HEATMAP_COLOR_MAP,
            mask=np.nonzero(padded_mat) is True,
            zorder=1,
        )
        ax.axhline(
            similarity_mat.shape[0],
            color="black",
            linewidth=axis_width,
            zorder=2,
        )
        ax.axhline(0, color="black", linewidth=axis_width, zorder=2)
        ax.axhline(
            padded_mat.shape[0], color="black", linewidth=axis_width, zorder=2
        )
        ax.set_xticks([])
        ax.set_yticks(
            np.arange(0.5, 0.5 + len(method_lst), 1), method_lst, rotation=0
        )
        ax.yaxis.set_minor_locator(mticker.NullLocator())
        ax.set_ylabel("method", rotation=90)
        fig.savefig(
            os.path.join(
                save_dirname,
                f"{region} pattern heatmap.svg",
            )
        )


plot_pattern_subject_heatmap_and_save(
    df=dataset,
    similarity_df=similarity_df,
    save_dirname=GEN_DIRNAME,
    figsize=(10, 10),
)
del plot_pattern_subject_heatmap_and_save


# %%
# Plot subject-subject barplot


def plot_subject_subject_barplot_and_save(
    similarity_df: pd.DataFrame,
    save_dirname: str,
    figsize: tuple[int, int] | None = None,
) -> None:
    similarity_df = similarity_df[
        similarity_df["matrix_type"] == "subject-subject"
    ].copy()
    DATA_TYPE_LST = ["Real", "Random"]
    STIMULATION_LST = ["Visual", "Auditory"]
    for similarity_method in similarity_df["similarity"].unique():
        sim_method_df = similarity_df
        for preprocess_method in similarity_df["modify_method"].unique():
            method_df = similarity_df[
                sim_method_df["modify_method"] == preprocess_method
            ]
            fig, ax = plt.subplots(figsize=figsize)
            match preprocess_method:
                case "None":
                    similarity_value_column = (
                        f"{similarity_method.upper()} Value"
                    )
                case _:
                    similarity_value_column = f"{similarity_method.upper()} ({preprocess_method}) Value"
            barplot_df = pd.DataFrame(
                columns=[similarity_value_column, "stimulation", "region"]
            )
            for data_type in DATA_TYPE_LST:
                for stimulation in STIMULATION_LST:
                    for _, row in method_df.iterrows():
                        mat: np.ndarray = row["matrix"]
                        axis_0_indices = row["axis_0_indices"]
                        axis_1_indices = row["axis_1_indices"]
                        region = row["region"]
                        axis_0_info = dict(
                            filter(
                                partial(
                                    verify_datatype_and_stimulation,
                                    data_type=data_type,
                                    stimulation=stimulation,
                                ),
                                axis_0_indices.items(),
                            )
                        )
                        axis_1_info = dict(
                            filter(
                                partial(
                                    verify_datatype_and_stimulation,
                                    data_type=data_type,
                                    stimulation=stimulation,
                                ),
                                axis_1_indices.items(),
                            )
                        )
                        del axis_0_indices, axis_1_indices
                        for axis_0_idx, axis_1_idx in product(
                            axis_0_info.keys(), axis_1_info.keys()
                        ):
                            new_sim_data = (
                                pd.Series(
                                    {
                                        similarity_value_column: mat[
                                            axis_0_idx, axis_1_idx
                                        ],
                                        "stimulation": f"{data_type} {stimulation}",
                                        "region": region,
                                    }
                                )
                                .to_frame()
                                .T
                            )
                            barplot_df = pd.concat(
                                [barplot_df, new_sim_data], ignore_index=True
                            )
            barplot_df.sort_values(by=["region", "stimulation"], inplace=True)
            barplot_df[similarity_value_column] = barplot_df[
                similarity_value_column
            ].astype(float)
            barplot_df["stimulation"] = barplot_df["stimulation"].astype(str)
            barplot_df["region"] = barplot_df["region"].astype(str)
            sns.barplot(
                x="region",
                y=similarity_value_column,
                hue="stimulation",
                data=barplot_df,
                ax=ax,
                ci=95,
                n_boot=10000,
                seed=42,
                errcolor="black",
                capsize=0.1,
            )
            for bar_container, label in zip(*ax.get_legend_handles_labels()):
                for rectangle in bar_container:
                    rectangle.set(
                        edgecolor="black", facecolor=GROUP_COLORS[label]
                    )
            bar_paris = [
                ((region, "Real Auditory"), (region, "Real Visual"))
                for region in barplot_df["region"].unique()
            ]
            annotator = Annotator(
                ax,
                pairs=bar_paris,
                plot="barplot",
                data=barplot_df,
                x="region",
                y=similarity_value_column,
                hue="stimulation",
            )
            annotator.configure(test="Mann-Whitney", text_format="star")
            annotator.apply_and_annotate()
            ax.get_legend().remove()
            fig.savefig(
                os.path.join(
                    save_dirname,
                    f"{similarity_method} {preprocess_method} barplot.svg",
                )
            )


plot_subject_subject_barplot_and_save(
    similarity_df=similarity_df, save_dirname=GEN_DIRNAME, figsize=(10, 10)
)
del plot_subject_subject_barplot_and_save


# %%


def get_kmeans_cluster_res(
    similarity_df: pd.DataFrame, save_dirname: str
) -> None:
    N_CLUSTERS = 2
    similarity_df = similarity_df[
        similarity_df["matrix_type"] == "subject-subject"
    ].copy()
    cluster_df_dict = {}
    for similarity_method in similarity_df["similarity"].unique():
        for preprocess_method in similarity_df["modify_method"].unique():
            method_similarity_df = similarity_df[
                (similarity_df["similarity"] == similarity_method)
                & (similarity_df["modify_method"] == preprocess_method)
            ]
            method = f"{similarity_method} ({preprocess_method})"
            cluster_df = pd.DataFrame(
                columns=[
                    "subject_idx",
                    f"{method} specific",
                    f"{method} nonspecific",
                    "is specific",
                ]
            )
            for region in method_similarity_df["region"].unique():
                structure = analysis_toolkit.dataset.get_structure(
                    region=region
                )
                match structure:
                    case "FFA":
                        specific_stimulation = "Visual"
                        non_specific_stimulation = "Auditory"
                    case "STG":
                        specific_stimulation = "Auditory"
                        non_specific_stimulation = "Visual"
                del structure
                region_similarity_df = method_similarity_df[
                    method_similarity_df["region"] == region
                ]
                similarity_mat = region_similarity_df["matrix"].iloc[0]
                axis_0_index_info = region_similarity_df[
                    "axis_0_indices"
                ].iloc[0]
                axis_1_index_info = region_similarity_df[
                    "axis_1_indices"
                ].iloc[0]
                specific_index_info_1 = dict(
                    filter(
                        partial(
                            verify_datatype_and_stimulation,
                            data_type="Real",
                            stimulation=specific_stimulation,
                        ),
                        axis_1_index_info.items(),
                    )
                )
                non_specific_index_info_1 = dict(
                    filter(
                        partial(
                            verify_datatype_and_stimulation,
                            data_type="Real",
                            stimulation=non_specific_stimulation,
                        ),
                        axis_1_index_info.items(),
                    )
                )
                del axis_1_index_info

                for axis_0_stimulation in ["Auditory", "Visual"]:
                    is_specific = (
                        1 if axis_0_stimulation == specific_stimulation else 0
                    )
                    real_axis_0_index_info = dict(
                        filter(
                            partial(
                                verify_datatype_and_stimulation,
                                data_type="Real",
                                stimulation=axis_0_stimulation,
                            ),
                            axis_0_index_info.items(),
                        )
                    )
                    for (
                        mat_idx_0,
                        subject_idx_info_0,
                    ) in real_axis_0_index_info.items():
                        subject_idx_0 = subject_idx_info_0.subject_idx
                        specific_mat_idx_lst = [
                            mat_idx_1
                            for mat_idx_1, subject_idx_info_1 in specific_index_info_1.items()
                            if subject_idx_info_1.subject_idx != subject_idx_0
                        ]
                        non_specific_mat_idx_lst = [
                            mat_idx_1
                            for mat_idx_1, subject_idx_info_1 in non_specific_index_info_1.items()
                            if subject_idx_info_1.subject_idx != subject_idx_0
                        ]
                        specific_similarity = similarity_mat[
                            np.ix_([mat_idx_0], specific_mat_idx_lst)
                        ]
                        non_specific_similarity = similarity_mat[
                            np.ix_([mat_idx_0], non_specific_mat_idx_lst)
                        ]
                        specific_similarity = np.mean(specific_similarity)
                        non_specific_similarity = np.mean(
                            non_specific_similarity
                        )
                        new_data = (
                            pd.Series(
                                {
                                    "subject_idx": subject_idx_0,
                                    f"{method} specific": specific_similarity,
                                    f"{method} nonspecific": non_specific_similarity,
                                    "is specific": is_specific,
                                }
                            )
                            .to_frame()
                            .T
                        )
                        cluster_df = pd.concat(
                            [cluster_df, new_data], ignore_index=True
                        )
            cluster_df_dict[method] = cluster_df
            res = analysis_toolkit.clustering.kmeans_cluster(
                data=cluster_df[
                    [f"{method} specific", f"{method} nonspecific"]
                ].values,
                n_clusters=N_CLUSTERS,
            )
            fig, axs = plt.subplots(1, 2, figsize=(12.8, 4.8))
            axs = axs.flatten()
            axs[0].scatter(
                cluster_df[f"{method} specific"],
                cluster_df[f"{method} nonspecific"],
                c=cluster_df["is specific"],
                cmap=ListedColormap(["Navy", "Red"], "Clusters"),
                s=10,
            )
            axs[0].set_xlabel("specific similarity")
            axs[0].set_ylabel("non-specific similarity")
            axs[0].set_title("True Clusters")
            axs[1].scatter(
                cluster_df[f"{method} specific"],
                cluster_df[f"{method} nonspecific"],
                c=res.labels,
                cmap=ListedColormap(["Red", "Navy"], "Clusters"),
                s=10,
            )
            axs[1].set_xlabel("specific similarity")
            axs[1].set_ylabel("non-specific similarity")
            axs[1].set_title("K-Means Clustering Result")
            plt.show(fig)
            fig.savefig(
                os.path.join(save_dirname, f"{method} K-Means cluster.svg")
            )
            acc = sklearn.metrics.accuracy_score(
                cluster_df["is specific"], res.labels
            )
            print(f"{method} Accuracy: {max(acc, 1-acc):.2%}")
            np.set_printoptions(precision=3)
            print(f"{method} center: {res.centers}")
            with open(
                os.path.join(save_dirname, "kmeans accuracy.txt"), "a"
            ) as f:
                f.write(f"{method} Accuracy: {max(acc, 1-acc)}\n")
                f.write(f"{method} center: {res.centers}\n")
    cluster_df = None
    for method, df in cluster_df_dict.items():
        if cluster_df is None:
            cluster_df = df
        else:
            cluster_df = pd.merge(
                cluster_df,
                df.loc[:, ~df.columns.isin(["is specific"])],
                on="subject_idx",
            )
    data = cluster_df.loc[
        :, ~cluster_df.columns.isin(["is specific", "subject_idx"])
    ]
    res = analysis_toolkit.clustering.kmeans_cluster(
        data=data.values, n_clusters=N_CLUSTERS
    )
    acc = sklearn.metrics.accuracy_score(cluster_df["is specific"], res.labels)
    print(f"Total Accuracy: {max(acc, 1-acc)}")
    with open(os.path.join(save_dirname, "kmeans accuracy.txt"), "a") as f:
        f.write(f"Total Accuracy: {max(acc, 1-acc)}\n")
        f.write(f"Total center: {res.centers}\n")


get_kmeans_cluster_res(similarity_df=similarity_df, save_dirname=GEN_DIRNAME)
# %%


def get_tree_model(similarity_df: pd.DataFrame, save_dirname: str) -> None:
    RANDOM_STATE = 42
    MIN_SAMPLES_LEAF = 3
    MAX_DEPTH = 2
    MAX_LEAF_NODES = None
    similarity_df = similarity_df[
        similarity_df["matrix_type"] == "subject-subject"
    ].copy()
    cluster_df_dict = {}
    for similarity_method in similarity_df["similarity"].unique():
        for preprocess_method in similarity_df["modify_method"].unique():
            method_similarity_df = similarity_df[
                (similarity_df["similarity"] == similarity_method)
                & (similarity_df["modify_method"] == preprocess_method)
            ]
            method = f"{similarity_method} ({preprocess_method})"
            cluster_df = pd.DataFrame(
                columns=[
                    "subject_idx",
                    f"{method} specific",
                    f"{method} nonspecific",
                    "is specific",
                ]
            )
            for region in method_similarity_df["region"].unique():
                structure = analysis_toolkit.dataset.get_structure(
                    region=region
                )
                match structure:
                    case "FFA":
                        specific_stimulation = "Visual"
                        non_specific_stimulation = "Auditory"
                    case "STG":
                        specific_stimulation = "Auditory"
                        non_specific_stimulation = "Visual"
                del structure
                region_similarity_df = method_similarity_df[
                    method_similarity_df["region"] == region
                ]
                similarity_mat = region_similarity_df["matrix"].iloc[0]
                axis_0_index_info = region_similarity_df[
                    "axis_0_indices"
                ].iloc[0]
                axis_1_index_info = region_similarity_df[
                    "axis_1_indices"
                ].iloc[0]
                specific_index_info_1 = dict(
                    filter(
                        partial(
                            verify_datatype_and_stimulation,
                            data_type="Real",
                            stimulation=specific_stimulation,
                        ),
                        axis_1_index_info.items(),
                    )
                )
                non_specific_index_info_1 = dict(
                    filter(
                        partial(
                            verify_datatype_and_stimulation,
                            data_type="Real",
                            stimulation=non_specific_stimulation,
                        ),
                        axis_1_index_info.items(),
                    )
                )
                del axis_1_index_info

                for axis_0_stimulation in ["Auditory", "Visual"]:
                    is_specific = (
                        1 if axis_0_stimulation == specific_stimulation else 0
                    )
                    real_axis_0_index_info = dict(
                        filter(
                            partial(
                                verify_datatype_and_stimulation,
                                data_type="Real",
                                stimulation=axis_0_stimulation,
                            ),
                            axis_0_index_info.items(),
                        )
                    )
                    for (
                        mat_idx_0,
                        subject_idx_info_0,
                    ) in real_axis_0_index_info.items():
                        subject_idx_0 = subject_idx_info_0.subject_idx
                        specific_mat_idx_lst = [
                            mat_idx_1
                            for mat_idx_1, subject_idx_info_1 in specific_index_info_1.items()
                            if subject_idx_info_1.subject_idx != subject_idx_0
                        ]
                        non_specific_mat_idx_lst = [
                            mat_idx_1
                            for mat_idx_1, subject_idx_info_1 in non_specific_index_info_1.items()
                            if subject_idx_info_1.subject_idx != subject_idx_0
                        ]
                        specific_similarity = similarity_mat[
                            np.ix_([mat_idx_0], specific_mat_idx_lst)
                        ]
                        non_specific_similarity = similarity_mat[
                            np.ix_([mat_idx_0], non_specific_mat_idx_lst)
                        ]
                        specific_similarity = np.mean(specific_similarity)
                        non_specific_similarity = np.mean(
                            non_specific_similarity
                        )
                        new_data = (
                            pd.Series(
                                {
                                    "subject_idx": subject_idx_0,
                                    f"{method} specific": specific_similarity,
                                    f"{method} nonspecific": non_specific_similarity,
                                    "is specific": is_specific,
                                }
                            )
                            .to_frame()
                            .T
                        )
                        cluster_df = pd.concat(
                            [cluster_df, new_data], ignore_index=True
                        )
            cluster_df_dict[method] = cluster_df
            model = DecisionTreeClassifier(
                random_state=RANDOM_STATE,
                min_samples_leaf=MIN_SAMPLES_LEAF,
                max_leaf_nodes=MAX_LEAF_NODES,
                max_depth=MAX_DEPTH,
            )
            x = cluster_df[[f"{method} specific", f"{method} nonspecific"]]
            y = cluster_df["is specific"]
            kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            scores = []
            for train_idx, test_idx in kf.split(x):
                x_train_fold, x_test_fold = x.iloc[train_idx], x.iloc[test_idx]
                y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]
                model.fit(x_train_fold, y_train_fold)
                y_pred = model.predict(x_test_fold)
                score = accuracy_score(y_test_fold, y_pred)
                scores.append(score)

            model.fit(x, y)
            y_pred = model.predict(x)
            fig, ax = plt.subplots(figsize=(30, 30))
            sklearn.tree.plot_tree(model, filled=True, ax=ax)
            fig.show()
            fig, ax = plt.subplots()
            ax: plt.Axes
            ax.barh(
                range(2), model.feature_importances_, align="center", color="c"
            )
            ax.set_yticks(
                range(2), ["specific similarity", "non-specific similarity"]
            )
            ax.set_xlabel("Feature importance")
            ax.set_ylabel("Feature")
            ax.yaxis.set_minor_locator(mticker.NullLocator())
            fig.show()
            fig, axs = plt.subplots(1, 2, figsize=(12.8, 4.8))
            axs = axs.flatten()
            axs[0].scatter(
                cluster_df[f"{method} specific"],
                cluster_df[f"{method} nonspecific"],
                c=cluster_df["is specific"],
                cmap=ListedColormap(["Navy", "Red"], "Clusters"),
                s=10,
            )
            axs[0].set_xlabel("specific similarity")
            axs[0].set_ylabel("non-specific similarity")
            axs[0].set_title("True Clusters")
            axs[1].scatter(
                cluster_df[f"{method} specific"],
                cluster_df[f"{method} nonspecific"],
                c=model.predict(
                    cluster_df[
                        [f"{method} specific", f"{method} nonspecific"]
                    ].values
                ),
                cmap=ListedColormap(["Navy", "Red"], "Clusters"),
                s=10,
            )
            axs[1].set_xlabel("specific similarity")
            axs[1].set_ylabel("non-specific similarity")
            axs[1].set_title("Decision Tree Result")
            plt.show(fig)
            fig.savefig(os.path.join(save_dirname, f"{method} tree.svg"))
            score = sklearn.metrics.accuracy_score(y, y_pred)
            print(f"{method} Accuracy: {score}")
            print(f"{method} Accurecy (K-Fold): {np.mean(scores)}")
            with open(
                os.path.join(save_dirname, "tree accuracy.txt"), "a"
            ) as f:
                f.write(f"{method} Accuracy: {score}\n")
    # cluster_df = None
    # for method, df in cluster_df_dict.items():
    #     if cluster_df is None:
    #         cluster_df = df
    #     else:
    #         cluster_df = pd.merge(
    #             cluster_df,
    #             df.loc[:, ~df.columns.isin(["is specific"])],
    #             on="subject_idx",
    #         )
    # x = cluster_df.loc[
    #     :, ~cluster_df.columns.isin(["is specific", "subject_idx"])
    # ]
    # y = cluster_df["is specific"]
    # kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    # scores = []

    # model = DecisionTreeClassifier(
    #     random_state=RANDOM_STATE,
    #     min_samples_leaf=MIN_SAMPLES_LEAF,
    #     max_leaf_nodes=MAX_LEAF_NODES,
    # )
    # for fold, (train_idx, test_idx) in enumerate(kf.split(x)):
    #     x_train_fold, x_test_fold = (
    #         x.iloc[train_idx],
    #         x.iloc[test_idx],
    #     )
    #     y_train_fold, y_test_fold = (
    #         y.iloc[train_idx],
    #         y.iloc[test_idx],
    #     )
    #     model.fit(x_train_fold, y_train_fold)
    #     y_pred = model.predict(x_test_fold)
    #     score = sklearn.metrics.accuracy_score(y_test_fold, y_pred)
    #     print(f"Fold Accuracy: {score}")
    #     scores.append(score)
    #     fig, ax = plt.subplots(figsize=(30, 30))
    #     sklearn.tree.plot_tree(model, filled=True, ax=ax)
    #     ax.set_title(f"Fold {fold}")
    #     fig.show()
    #     fig, ax = plt.subplots()
    #     ax.barh(range(len(x.columns)), model.feature_importances_, align="center")
    #     ax.set_yticks(range(len(x.columns)), x.columns)
    #     ax.set_xlabel("Feature importance")
    #     ax.set_ylabel("Feature")
    #     fig.show()

    # score = np.mean(scores)
    # print(f"Test Accurecy: {score}")
    # with open(os.path.join(save_dirname, "kmeans accuracy.txt"), "a") as f:
    #     f.write(f"Total Accuracy: {score}\n")


get_tree_model(similarity_df=similarity_df, save_dirname=GEN_DIRNAME)
# %%
