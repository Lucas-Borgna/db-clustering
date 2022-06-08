import pandas as pd
import uproot
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mplhep as hep

hep.style.use("CMS")

from tqdm import tqdm

import hdbscan

from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.proportion import proportion_confint
import copy

from notebooks.primaryvertexingtools import PrimaryVertexing, PerformanceMetrics

from notebooks.primaryvertexingtools import remove_nans, create_pv_truth_labels
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report


class PrimaryVertexing(object):
    def __init__(self, object):
        self.fh_bins = object["fh_bins"]
        self.eta_bins = np.array([0.0, 0.7, 1.0, 1.2, 1.6, 2.0, 2.4])
        self.deltaz_bins = np.array([0.0, 0.4, 0.6, 0.76, 1.0, 1.7, 2.2, 0.0])
        self.test_run = object["test_run"]
        self.trk = object["track_data"]
        self.mc = object["truth_data"]
        if self.test_run:
            self.trk = self.trk.query("entry<=10").copy()
            self.mc = self.mc.query("entry<=10").copy()
        
        print("Initialized Primary Vertexing setup")

    def z0_reco_fh_bin(self, max_index: int) -> float:
        "Function returns the reconstructed z0 value for fast histo based on the maximum bin"
        bin_edges = self.fh_bins
        half_bin_width = 0.5 * (bin_edges[1] - bin_edges[0])
        lowest_bin = bin_edges[0]
        highest_bin = bin_edges[-1]
        nbins = bin_edges.shape[0]

        z0 = (
            lowest_bin
            + (highest_bin - lowest_bin) * (max_index / nbins)
            + half_bin_width
        )

        return z0

    def fh_pv_association(
        self, distance_from_pv: np.array, eta: np.array
    ) -> np.array(np.float32):

        eta_bin = np.digitize(np.abs(eta), self.eta_bins)
        assoc = distance_from_pv < self.deltaz_bins[eta_bin]

        return np.array(assoc, dtype=np.float32)

    def fh(
        self, z0: np.array, pt: np.array, eta: np.array, bin_edges: np.array
    ) -> pd.Series:
        "Runs fast histo on a single event"

        histo = np.histogram(z0, bins=bin_edges, weights=pt)[0]

        histo = np.convolve(histo, [1, 1, 1], mode="same")
        max_index = np.argmax(histo)

        z0_pv = self.z0_reco_fh_bin(max_index)
        z0_array = z0_pv * np.ones(z0.shape[0], dtype=np.float32)

        return pd.Series(z0_array)

    def run_fh(self):
        "Runs fast histo on all events"
        bin_edges = self.fh_bins
        pv_fh = self.trk.groupby(level=0).progress_apply(
            lambda x: self.fh(x["trk_z0"], x["trk_pt"], x["trk_eta"], bin_edges)
        )

        self.trk["z0_reco_fh"] = pv_fh

        self.z0_reco_fh = self.trk.groupby(level=0)["z0_reco_fh"].first().values

        self.trk["distance_from_pv_fh"] = np.abs(
            self.trk["z0_reco_fh"] - self.trk["trk_z0"]
        )

        self.trk["trk_pv_assoc_fh"] = self.fh_pv_association(
            self.trk["distance_from_pv_fh"], self.trk["trk_eta"]
        )
        self.fh_classification_metrics = self.trk_vertex_association(
            self.trk["is_pv"].values, self.trk["trk_pv_assoc_fh"].values
        )

        print(f"Ran Fast Histo")

    def hdbscan(
        self, z0: np.array, pt: pd.Series, eps: float = 0.08, minPts: int = 2
    ) -> pd.Series:

        _df = pd.DataFrame({})
        _df["z0"] = z0
        _df["pt"] = pt

        db_clustering = hdbscan.HDBSCAN(min_samples=minPts,cluster_selection_epsilon=0.08).fit(
            _df["z0"].values.reshape(-1, 1)
        )

        _df["db_label"] = db_clustering.labels_

        # Negative labels correspond to noise points, so floor pt 0 so they don't become the PV
        _df.loc[_df["db_label"] < 0, "pt"] = 0

        # Determine which DBSCAN label corresponds to the primary vertex.
        pv_label = (
            _df.groupby(["db_label"])["pt"].sum().sort_values(ascending=False).index[0]
        )

        _df["db_pv_label"] = 0

        _df.loc[_df["db_label"] == pv_label, "db_pv_label"] = 1

        z0_reco = np.median(_df.loc[_df["db_pv_label"] == 1, "z0"])
        _df["z0_reco_db"] = z0_reco

        return _df[["db_pv_label", "z0_reco_db"]]

    def run_hdbscan(self, eps: float = 0.08, minPts: int = 2):

        pv_hdbscan = self.trk.groupby(level=0).progress_apply(
            lambda x: self.hdbscan(x["trk_z0"], x["trk_pt"], eps, minPts)
        )

        self.trk["z0_reco_db"] = pv_hdbscan["z0_reco_db"]
        self.z0_reco_db = self.trk.groupby(level=0)["z0_reco_db"].first().values

        self.trk["trk_pv_assoc_db"] = pv_hdbscan["db_pv_label"]

        print(f"Ran DBSCAN with eps: {eps}, minPts: {minPts}")

    def trk_vertex_association(self, y_true, y_pred) -> dict:
        """Calculates Binary Classification metrics
        Returns:
            dict: classification metrics
        """

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        TPR = tp / (tp + fn)

        FPR = fp / (tn + fp)

        AUC = roc_auc_score(y_true, y_pred)

        return {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "TPR": TPR,
            "FPR": FPR,
            "AUC": AUC,
        }
    
class PerformanceMetrics(object):
    def __init__(self, object):
        self.trk = object.trk
        self.z0_reco_fh = object.z0_reco_fh
        self.z0_reco_db = object.z0_reco_db
        self.mc = object.mc

        self.z0_gen = self.mc["pv_MC"].values
        self.res_fh = self.z0_gen - self.z0_reco_fh
        self.delta = 0.1
        self.profile_bins = np.arange(-15, 16, 1)

        self.positive_index = (
            self.mc[(self.mc["pv_MC"] > 0)].index.get_level_values(0).values
        )
        self.negative_index = (
            self.mc[(self.mc["pv_MC"] < 0)].index.get_level_values(0).values
        )

        self.trk["event_number"] = self.trk.index.get_level_values(0).values

        self.correct_bias_fh()
        self.correct_bias_db()

    def correct_bias_fh(self):
        self.z0_reco_fh_cor = self.z0_reco_fh.copy()

        mask = self.z0_gen > 0
        bias = np.median((self.z0_gen[mask] - self.z0_reco_fh[mask]))
        print(bias)
        self.z0_reco_fh_cor[mask] = self.z0_reco_fh_cor[mask] + bias
        self.trk["z0_reco_fh_cor"] = self.trk["z0_reco_fh"] + bias

    def correct_bias_db(self):
        self.z0_reco_db_cor = self.z0_reco_db.copy()
        self.trk["z0_reco_db_cor"] = self.trk["z0_reco_db"]
        positive_mask = self.z0_gen > 0
        negative_mask = self.z0_gen < 0

        positive_bias = np.median(
            (self.z0_gen[positive_mask] - self.z0_reco_db[positive_mask])
        )

        print("positive bias", positive_bias)
        self.z0_reco_db_cor[positive_mask] = (
            self.z0_reco_db_cor[positive_mask] + positive_bias
        )
        df_pos_mask = self.trk["event_number"].isin(self.positive_index)

        self.trk.loc[df_pos_mask, "z0_reco_db_cor"] = (
            self.trk.loc[df_pos_mask, "z0_reco_db_cor"] + positive_bias
        )

        negative_bias = np.median(
            (self.z0_gen[negative_mask] - self.z0_reco_db[negative_mask])
        )
        print("negative bias", negative_bias)

        self.z0_reco_db_cor[negative_mask] = (
            self.z0_reco_db_cor[negative_mask] + negative_bias
        )

        df_neg_mask = self.trk["event_number"].isin(self.negative_index)
        self.trk.loc[df_neg_mask, "z0_reco_db_cor"] = (
            self.trk.loc[df_neg_mask, "z0_reco_db_cor"] + negative_bias
        )

    def bin_width_error(self, bin_edges):
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

    def pv_efficiency(self, res: np.array, display=False):

        diff = np.abs(res)

        Npass = (diff <= self.delta).sum()
        Ntotal = diff.shape[0]

        self.pv_eff = 100 * Npass / Ntotal
        if display:
            print(f"{round(self.pv_eff, 2)} %")

    def pv_resolution(
        self,
        z0_gen: np.array,
        z0_reco: np.array,
        bins: np.array = None,
        label: str = None,
        title: str = "Primary Vertex Resolution",
        xlim: list = None,
    ):
        if bins is None:
            bins = self.profile_bins

        res = z0_gen - z0_reco

        q = np.percentile((res), [32, 50, 68])
        q_w = round(q[2] - q[0], 3)
        RMS = round(np.sqrt(np.mean((res) ** 2)), 2)
        median = round(np.median(res), 3)

        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        full_label = (
            label
            + "RMS: "
            + str(RMS)
            + "\nquartile width: "
            + str(q_w)
            + "\nmedian = "
            + str(median)
        )

        h, be, _ = ax.hist(
            res,
            bins=bins,
            range=(-5, 5),
            histtype="step",
            label=full_label,
            lw=2,
        )
        ax.axvline(median, ls="--", lw=2, color="grey", label="median")
        ax.legend()
        ax.grid(True)
        ax.set_xlabel(r"$z_{0}^{PV, Gen} - z_{0}^{PV, Reco}$ [cm]", ha="right", x=1)
        ax.set_ylabel("Events", ha="right", y=1)
        ax.set_title(title)
        ax.set_yscale("log")
        if xlim is not None:
            ax.set_xlim(xlim)
        return {"hist": h, "bins": be, "label": full_label}

    def res_vs_z0(
        self,
        res: np.array,
        z0_gen: np.array,
        bins: tuple = (50, 50),
        hrange: tuple = ((-15, 15), (-5, 5)),
        cmap: str = "magma",
        dolog: bool = True,
    ):
        #         if res is None:
        #             res = self.res_fh
        #         if z0_gen is None:
        #             z0_gen = self.z0_reco_fh

        y = res
        x = z0_gen
        if dolog:
            _ = plt.hist2d(x, y, bins=bins, range=hrange, norm=LogNorm(), cmap=cmap)
        else:
            _ = plt.hist2d(x, y, bins=bins, range=hrange, cmap=cmap)

        plt.xlabel(r"$z_0^{PV, Gen}$ [cm]")
        plt.ylabel(r"$z_0^{PV, Gen} - z_0^{PV, Reco}$ [cm]")
        plt.colorbar()

    def plot_pv_resolution_z0(
        self, z0_reco: np.array = None, z0_gen: np.array = None, bins=None, label=None
    ):
        if bins is None:
            bins = self.profile_bins
        if z0_gen is None:
            z0_gen = self.z0_gen
        if z0_reco is None:
            z0_reco = self.z0_reco_fh

        n_bins = bins.shape[0] - 1
        mean_resolution = np.zeros(n_bins)
        error_resolution = np.zeros(n_bins)
        correctly_reconstructed_mask = np.abs(z0_gen - z0_reco) < self.delta

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
            x,
            mean_resolution,
            xerr=error_x,
            yerr=error_resolution,
            fmt="k+",
            label=label,
        )
        plt.xlabel(r"$z_0^{PV, Gen}$ [cm] ")
        plt.ylabel(r"$|z_0^{PV, Gen} - z_0^{PV, Reco}|$ [cm]")
        plt.ylim([0.0, 0.1])
        plt.legend()

        return {
            "x": x,
            "mean_res": mean_resolution,
            "xerr": error_x,
            "yerr": error_resolution,
            "label": label,
            "bins": bins,
            "n_bins": n_bins,
        }

    def plot_pv_efficiency_z0(
        self,
        z0_gen: np.array = None,
        z0_reco: np.array = None,
        bins=None,
        label=None,
        xlim: list = [-15, 15],
    ):

        if bins is None:
            bins = self.profile_bins
        n_bins = bins.shape[0] - 1
        if z0_gen is None:
            z0_gen = self.z0_gen
        if z0_reco is None:
            z0_reco = self.z0_reco_fh

        x = 0.5 * (bins[1:] + bins[:-1])
        error_x = self.bin_width_error(bins)

        results_df = pd.DataFrame({})

        eff = np.zeros(n_bins)
        error_eff = np.zeros((n_bins, 2))

        reconstructed_pvs = np.zeros(n_bins)
        total_pvs = np.zeros(n_bins)

        for i, _ in enumerate(bins):
            if i == n_bins:
                break

            bin_mask = (z0_gen > bins[i]) & (z0_gen < bins[i + 1])

            pv_reconstructed_mask = np.abs(z0_gen - z0_reco) < self.delta

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
        plt.xlabel(r"$z_0^{PV, Gen}$ [cm]")
        plt.legend()

        return results_df

    def trk_vertex_association(self, y_true, y_pred) -> dict:
        """Calculates Binary Classification metrics
        Returns:
            dict: classification metrics
        """

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        TPR = tp / (tp + fn)

        FPR = fp / (tn + fp)

        AUC = roc_auc_score(y_true, y_pred)

        return {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "TPR": TPR,
            "FPR": FPR,
            "AUC": AUC,
        }

def main():
    
    print("h")
    
    
    storage = '/media/lucas/MicroSD/l1_nnt/'
#     storage = "/Volumes/ExternalSSD/track/l1_nnt/"
#     tp_original = pd.read_pickle(storage + "tp.pkl")
    trk = pd.read_pickle(storage + "trk_25k.pkl")
    mc = pd.read_pickle(storage + "mc_25k.pkl")
    
    trk = remove_nans(trk, feature="trk_eta")
    trk = create_pv_truth_labels(trk, truth_label="trk_fake", truth_label_out="is_pv")
    
    be = np.linspace(-15, 15, 256)
    
    pv_setup = {
    "fh_bins": np.linspace(-15, 15, 256),
    "truth_data": mc,
    "track_data": trk,
    "test_run": False,
    }
    
    pv = PrimaryVertexing(pv_setup)
    
    pv.run_fh()
    
    pv.run_hdbscan()
    
    pm = PerformanceMetrics(pv)
        
    pm.pv_efficiency((pm.z0_gen - pm.z0_reco_db_cor), display=True)
    

    
    
    
if __name__ == "__main__":
    
    main()