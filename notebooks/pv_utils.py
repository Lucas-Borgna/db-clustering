import numpy as np
from numpy import core
import pandas as pd
from sklearn.cluster import DBSCAN
from statsmodels.stats.proportion import proportion_confint
import matplotlib.pyplot as plt
from scipy.stats import iqr
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

import uproot

from tqdm import tqdm

tqdm.pandas()


if int(uproot.__version__[0]) < 4:
    raise ImportError("This will only work with uproot version 4")


def res_function(eta):
    res = 0.1 + 0.2 * eta ** 2
    return res


def load_data(filename: str, data_type: str) -> pd.DataFrame:
    "load truth data tracks from file"

    f = uproot.open(filename)

    if data_type == "truth":
        return f["L1TrackNtuple/eventTree;1"].arrays(library="pd")[1]
    elif data_type == "track":
        return f["L1TrackNtuple/eventTree;1"].arrays(library="pd")[0]
    else:
        raise ValueError("data_type needs to be either 'truth' or 'track'")


def data_load(filename: str):
    "load both truth data and track data from file"

    f = uproot.open(filename)

    dfs = f["L1TrackNtuple/eventTree;1"].arrays(library="pd")

    tp = dfs[1]
    trk = dfs[0]

    return tp, trk


def truth_pv_z0(df: pd.DataFrame) -> pd.DataFrame:
    "Returns the z0 values of the primary vertex at truth level (z_gen)"

    mask_pv = df["tp_eventid"] == 0
    mask_no_sv = np.abs(df["tp_d0"]) < 0.01

    z0_gen = df.loc[mask_pv & mask_no_sv].groupby(level=0)["tp_z0"].mean()

    return z0_gen


def fast_histo(z0: np.array, pt: np.array, bin_edges: np.array) -> pd.Series:
    "DEPRECATED!!!!! (only kept for old results and debugging). Event-level Fast Histo implementation to return the reconstructed primary vertex"

    histo = np.histogram(z0, bins=bin_edges, weights=pt)[0]

    max_idx = np.argmax(histo)

    lower_bin_bound = bin_edges[max_idx]
    upper_bin_bound = bin_edges[max_idx + 1]

    in_max_bin_mask = (z0 > lower_bin_bound) & (z0 < upper_bin_bound)

    z0_reco = z0[in_max_bin_mask].mean()

    return z0_reco


def run_fast_histo(
    df: pd.DataFrame,
    bin_edges: np.array,
    z0_column: str = "trk_z0",
    pt_column: str = "trk_pt",
) -> pd.DataFrame:
    "DEPRECATED!!!!! (only kept for old results and debugging). Runs fast histo on all of the available events"

    z0_fast_histo = df.groupby(level=0).apply(
        lambda x: fast_histo(x[z0_column], x[pt_column], bin_edges)
    )

    return z0_fast_histo


def pv_dbscan(
    z0: np.array, pt: np.array, eps: float, minPts: int, remove_noise: bool = False
) -> pd.Series:
    "DEPRECATED!!!!! (only kept for old results and debugging). Event-level DBSCAN implementation to return the reconstructed primary vertex"

    results = pd.DataFrame({})

    results["z0"] = z0
    results["pt"] = pt

    db_clustering = DBSCAN(eps=eps, min_samples=minPts).fit(z0.values.reshape(-1, 1))

    results["dbscan_labels"] = db_clustering.labels_

    if remove_noise:
        results.loc[results["dbscan_labels"] == -1, "pt"] = 0

    mean_z0_dbscan = results.groupby(["dbscan_labels"]).agg(
        {"dbscan_labels": "count", "z0": "median", "pt": "sum"}
    )

    return mean_z0_dbscan.sort_values(by=["pt"], ascending=False).iloc[0, :]


def run_dbscan(
    df: pd.DataFrame,
    z0_column: str = "trk_z0",
    pt_column: str = "trk_pt",
    eps: float = 0.08,
    minPts: int = 2,
    remove_noise: bool = False,
) -> pd.DataFrame:
    """DEPRECATED!!!!! (only kept for old results and debugging). run dbscan"""

    z0_dbscan = df.groupby(level=0).apply(
        lambda x: pv_dbscan(x[z0_column], x[pt_column], eps, minPts, remove_noise)
    )

    return z0_dbscan


def primary_vertex_efficiency(
    z0_gen: np.array,
    z0_reco: np.array,
    delta: float = 0.1,
) -> float:
    "Returns the primary vertex reconstruction efficiency, which is dependent on the resolution (delta)"

    pv_matched_mask = np.abs(z0_gen - z0_reco) < delta

    total_pvs = z0_gen.shape[0]

    reconstructed_pvs = z0_gen[pv_matched_mask].shape[0]

    efficiency = 100 * (reconstructed_pvs / total_pvs)

    ci_low, ci_upp = proportion_confint(reconstructed_pvs, total_pvs, method="beta")
    ci_low = efficiency - ci_low * 100
    ci_upp = ci_upp * 100 - efficiency
    return efficiency, ci_low, ci_upp


def bin_width_error(bin_edges):
    """
    Determines the horizontal (x) error of a bin  by calculating half the bin size
    :param bin_edges:
    :return: xerr array containing the absolute magnitude of the error in x
    """
    # determines the error in a bin by +/- half of the bin_width
    xerr = []
    for k in range(len(bin_edges)):
        if k != (len(bin_edges) - 1):
            x1 = bin_edges[k]
            x2 = bin_edges[k + 1]
            bin_error = (x2 - x1) / 2
            xerr.append(bin_error)
    xerr = np.asarray(xerr)
    return xerr


def plot_pv_efficiency_z0(
    z0_gen: np.array,
    z0_reco: np.array,
    bin_edges: np.array,
    delta: float = 0.1,
    label: str = "none",
    xlim: list = [-15, 15],
) -> dict:
    """plots the efficiency of the primary vertex as a function of z0"""

    n_bins = bin_edges.shape[0] - 1

    x = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    error_x = bin_width_error(bin_edges)

    results_df = pd.DataFrame({})

    eff = np.zeros(n_bins)
    error_eff = np.zeros((n_bins, 2))

    reconstructed_pvs = np.zeros(n_bins)
    total_pvs = np.zeros(n_bins)

    for i, _ in enumerate(bin_edges):
        if i == n_bins:
            break

        bin_mask = (z0_gen > bin_edges[i]) & (z0_gen < bin_edges[i + 1])

        pv_reconstructed_mask = np.abs(z0_gen - z0_reco) < delta

        total_pv = z0_gen[bin_mask].shape[0]

        reconstructed_pv = z0_gen[bin_mask & pv_reconstructed_mask].shape[0]
        try:
            eff[i] = reconstructed_pv / total_pv
        except:
            eff[i] = 1
        reconstructed_pvs[i] = reconstructed_pv
        total_pvs[i] = total_pv

    ci_low, ci_upp = proportion_confint(reconstructed_pvs, total_pvs, method="beta")

    results_df["x"] = x
    results_df["error_x"] = error_x
    results_df["eff"] = eff
    results_df["ci_low"] = ci_low
    results_df["ci_upp"] = ci_upp
    results_df["lower_error"] = results_df["eff"] - results_df["ci_low"]
    results_df["upper_error"] = results_df["ci_upp"] - results_df["eff"]

    plt.errorbar(
        x,
        eff,
        xerr=error_x,
        yerr=[results_df["lower_error"].values, results_df["upper_error"].values],
        ls="none",
        label=label,
    )
    plt.xlim(xlim)
    plt.ylim(0, 1.1)
    plt.ylabel("Efficiency")
    plt.xlabel(r"$z_{gen}$")
    plt.legend()

    return results_df


def plot_pv_resolution_z0(
    z0_gen: np.array,
    z0_reco: np.array,
    bins: np.array,
    ylim: list = [0.0, 0.1],
    delta: float = 0.1,
    label: str = "None",
) -> dict:
    """Plots the resolution of the primary vertex as a function of z0"""

    n_bins = bins.shape[0] - 1
    mean_resolution = np.zeros(n_bins)
    error_resolution = np.zeros(n_bins)

    correctly_reconstructed_mask = np.abs(z0_gen - z0_reco) < delta

    for i, _ in enumerate(bins):
        if i == n_bins:
            break

        in_bin_mask = pd.Series((z0_gen > bins[i]) & (z0_gen < bins[i + 1]))
        mask = in_bin_mask & correctly_reconstructed_mask

        resolution = np.abs(z0_gen[mask] - z0_reco[mask])

        mean_resolution[i] = np.mean(resolution)
        error_resolution[i] = np.std(resolution)

    x = 0.5 * (bins[1:] + bins[:-1])
    error_x = 0.5 * (bins[1:] - bins[:-1])

    plt.errorbar(
        x, mean_resolution, xerr=error_x, yerr=error_resolution, fmt="k+", label=label
    )
    plt.xlabel(r"$z_{gen}$")
    plt.ylabel(r"$|z_{gen} - z_{reco}|$ [cm]")
    plt.ylim(ylim)
    plt.legend()

    return {
        "x": x,
        "mean_resolution": mean_resolution,
        "xerr": error_x,
        "yerr": error_resolution,
        "bins": bins,
    }


def fast_histo_event(z0: np.array, pt: np.array, bin_edges: np.array) -> pd.Series:
    "Event-level Fast Histo implementation to return the reconstructed primary vertex"

    histo = np.histogram(z0, bins=bin_edges, weights=pt)[0]

    histo_c = np.convolve(histo, [1, 1, 1], mode="same")

    labels = np.zeros(len(z0), dtype=int)

    max_idx = np.argmax(histo)
    max_idx_c = np.argmax(histo_c)

    lower_bin_bound = bin_edges[max_idx_c]
    upper_bin_bound = bin_edges[max_idx_c + 1]

    global max_idx_list
    global max_idx_list_c
    max_idx_list.append(max_idx)
    max_idx_list_c.append(max_idx_c)


    in_max_bin_mask = (z0 > lower_bin_bound) & (z0 <= upper_bin_bound)
    labels[in_max_bin_mask] = 1

    return labels


def run_pv_fast_histo(
    df: pd.DataFrame,
    bin_edges: np.array,
    z0_column: str = "trk_z0",
    pt_column: str = "trk_pt",
) -> np.array:
    """Runs FastHisto over all of the events.

    Args:
        df (pd.DataFrame): Dataframe containing multiple events
        bin_edges (np.array): bin edges to be used for FastHisto
        z0_column (str, optional): column name containing track's z0 position. Defaults to "trk_z0".
        pt_column (str, optional): column name containing tracks pt. Defaults to "trk_pt".

    Returns:
        np.array: Binary labels indicating if the track belongs to the primary vertex.
    """
    pv_fh = df.groupby(level=0).progress_apply(
        lambda x: fast_histo_event(x[z0_column], x[pt_column], bin_edges)
    )

    pv_fh = pd.DataFrame(pv_fh, columns=["fh_label"])

    pv_fh = pv_fh.explode("fh_label")

    return pv_fh.values


def pv_dbscan_event(
    z0: pd.Series, pt: pd.Series, eps: float = 0.08, minPts: int = 2
) -> pd.Series:
    """Runs DBSCAN over a signle event. Returns a binary outcome determining if the track belongs to the primary vertex.

    Args:
        z0 (pd.Series): z0 position of the track
        pt (pd.Series): pt of the track
        eps (float, optional): eps hyperparameter. Defaults to 0.08.
        minPts (int, optional): minPts hyperparameter. Defaults to 2.

    Returns:
        pd.Series: labels to indicate if track belongs to primary vertex
    """

    _df = pd.DataFrame({})
    _df["z0"] = z0
    _df["pt"] = pt

    db_clustering = DBSCAN(eps=eps, min_samples=minPts).fit(
        _df["z0"].values.reshape(-1, 1)
    )

    _df["db_label"] = db_clustering.labels_

    # -1 labels correspond to noise points, so floor these to 0 pt so they never become the PV.
    _df.loc[_df["db_label"] == -1, "pt"] = 0

    # Determine which DBSCAN label corresponds to the primary vertex.
    pv_label = (
        _df.groupby(["db_label"])["pt"].sum().sort_values(ascending=False).index[0]
    )

    _df["db_pv_label"] = 0

    _df.loc[_df["db_label"] == pv_label, "db_pv_label"] = 1

    return _df["db_pv_label"]


def run_pv_dbscan(
    df: pd.DataFrame,
    z0_column: str = "trk_z0",
    pt_column: str = "trk_pt",
    eps: float = 0.08,
    minPts: int = 2,
) -> np.array:
    """Runs DBSCAN over the whole dataset (multiple events).
    The algorithm will predict for each track in the event whether it belongs to the PV or not.

    Args:
        df (pd.DataFrame): dataframe containing trk z0 and pt information for dbscan to use
        z0_column (str, optional): column of the dataframe where the z0 information is stored. Defaults to "trk_z0".
        pt_column (str, optional): column of the dataframe where the pt information is stored. Defaults to "trk_pt".
        eps (float, optional): eps hyperparameter of DBSCAN. Defaults to 0.08.
        minPts (int, optional): minPts hyperparameter of DBSCAN. Defaults to 2.

    Returns:
        np.array: Array containing the predicted labels by DBSCAN
    """

    pv_dbscan = df.groupby(level=0).progress_apply(
        lambda x: pv_dbscan_event(x[z0_column], x[pt_column], eps, minPts)
    )

    return pv_dbscan.values


def pv_z0_reco(
    df: pd.DataFrame, reco_label: str = "fh_label", z0_label: str = "trk_z0"
) -> pd.DataFrame:
    """Returns the z0 of the primary vertex for a given reconstruction algorithm (fasthisto or dbscan)

    Args:
        df (pd.DataFrame): dataframe containing the labels of the reconstruction algorithm and the trk z0
        reco_label (str, optional): column of the dataframe containing the labels. Defaults to "fh_label".
        z0_label (str, optional): column of the dataframe that containes the z0 information. Defaults to "trk_z0".

    Returns:
        pd.DataFrame: Dataframe containing the reconstructed PV z0 for each event
    """

    return (
        df.loc[df[reco_label] == 1].groupby(["entry"])[z0_label].median().reset_index()
    )


def trk_vertex_association(
    df: pd.DataFrame, true_label: str = "trk_fake", pred_label: str = "fh_label"
) -> dict:
    """Calculates Binary Classification metrics

    Args:
        df (pd.DataFrame): dataframe containing true and predicted labels
        true_label (str, optional): column name of true label. Defaults to "trk_fake".
        pred_label (str, optional): column name of predicted labels. Defaults to "fh_label".

    Returns:
        dict: classification metrics
    """

    tn, fp, fn, tp = confusion_matrix(
        df[true_label].values, df[pred_label].values
    ).ravel()

    TPR = tp / (tp + fn)

    FPR = fp / (tn + fp)

    AUC = roc_auc_score(df[true_label].values, df[pred_label].values)

    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp, "TPR": TPR, "FPR": FPR, "AUC": AUC}


def make_confusion_matrix(
    cf,
    group_names=None,
    categories="auto",
    count=True,
    percent=True,
    cbar=True,
    xyticks=True,
    xyplotlabels=True,
    sum_stats=True,
    figsize=None,
    cmap="Blues",
    title=None,
):
    """
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html

    title:         Title for the heatmap. Default is None.
    """

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ["" for i in range(cf.size)]

    if group_names and len(group_names) == cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = [
            "{0:.2%}".format(value) for value in cf.flatten() / np.sum(cf)
        ]
    else:
        group_percentages = blanks

    box_labels = [
        f"{v1}{v2}{v3}".strip()
        for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)
    ]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        # Accuracy is sum of diagonal divided by total observations
        accuracy = np.trace(cf) / float(np.sum(cf))

        # if it is a binary confusion matrix, show some more stats
        if len(cf) == 2:
            # Metrics for Binary Confusion Matrices
            precision = cf[1, 1] / sum(cf[:, 1])
            recall = cf[1, 1] / sum(cf[1, :])
            f1_score = 2 * precision * recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy, precision, recall, f1_score
            )
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize == None:
        # Get default figure size if not set
        figsize = plt.rcParams.get("figure.figsize")

    if xyticks == False:
        # Do not show categories if xyticks is False
        categories = False

    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(
        cf,
        annot=box_labels,
        fmt="",
        cmap=cmap,
        cbar=cbar,
        xticklabels=categories,
        yticklabels=categories,
    )

    if xyplotlabels:
        plt.ylabel("True label")
        plt.xlabel("Predicted label" + stats_text)
    else:
        plt.xlabel(stats_text)

    if title:
        plt.title(title)
