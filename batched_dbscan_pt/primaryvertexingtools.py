import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.cluster import DBSCAN
import seaborn as sns
import matplotlib.pyplot as plt
import mplhep as hep
import itertools
import math

hep.style.use("CMS")
from statsmodels.stats.proportion import proportion_confint
from sklearn.metrics import confusion_matrix, roc_auc_score, auc, classification_report, roc_curve
import copy
import pickle
from matplotlib.colors import LogNorm

import sys

np.random.seed(42)

np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)
# sys.path.append("../")
# from batched_dbscan.batched_dbscan import BatchedDBSCAN


tqdm.pandas()


class PrimaryVertexing(object):
    def __init__(self, object):
        self.fh_bins = object["fh_bins"]
        self.eta_bins = np.array([0.0, 0.7, 1.0, 1.2, 1.6, 2.0, 2.4])
        self.deltaz_bins = np.array([0.0, 0.4, 0.6, 0.76, 1.0, 1.7, 2.2, 0.0])
        self.test_run = object["test_run"]
        self.trk = object["track_data"]
        self.mc = object["truth_data"]

        if self.test_run:
            if "nevents_test" not in object.keys():
                self.nevents_test = 1000
            else:
                self.nevents_test = object["nevents_test"]
            self.trk = self.trk.query(f"entry<={self.nevents_test}").copy()
            self.mc = self.mc.query(f"entry<={self.nevents_test}").copy()
        self.nevents = self.trk.index[-1][0]
        self.event_number = self.trk.reset_index()["entry"].values
        self.track_number = self.trk.reset_index()["subentry"].values
        self.trk["event_number"] = self.event_number
        self.trk["track_number"] = self.track_number
        self.fh_complete = False
        self.db_complete = False
        self.bdb_complete = False

        self.rank_by_pt = object["rank_by_pt"]
        if self.rank_by_pt:
            if "rank_limit_n" not in object.keys():
                self.rank_limit_n = 20
            else:
                self.rank_limit_n = object["rank_limit_n"]

        if "use_classifier_cut" in object.keys():
            self.use_classifier_cut = object["use_classifier_cut"]
            if "classifier_threshold" not in object.keys():
                self.classifier_threshold = 0.2
            else:
                self.classifier_threshold = object["classifier_threshold"]
            self.clf_filepath = object["clf_filepath"]
            self.clf = pickle.load(open(self.clf_filepath, "rb"))
        else:
            self.use_classifier_cut = False

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

    def plot_pv(
        self,
        n: int = 0,
        nbins: int = 10,
        delta: float = 0.15,
        figsize: tuple = (8, 8),
        fontsize: int = 14,
    ):

        _df = self.trk.query(f"entry=={n}")
        pv_truth = self.mc.query(f"entry=={n}")["pv_MC"].values
        z0_pv = _df.loc[_df["trk_pv_assoc_db"] == 1, "trk_z0"]
        z0_mean = np.mean(z0_pv)
        z0_median = np.median(z0_pv)
        z0_count = z0_pv.shape[0]
        plt.figure(figsize=figsize)
        plt.title(f"number of tracks = {z0_count}", fontsize=fontsize)
        h, _, _ = plt.hist(z0_pv, bins=nbins, histtype="step")
        plt.axvline(z0_mean, ls="--", lw=2, color="grey", label="mean")
        plt.axvline(z0_median, ls="--", lw=2, color="indigo", label="median")
        plt.axvline(pv_truth, ls="--", lw=2, color="green", label="truth")
        plt.fill_betweenx(
            np.linspace(0, max(h), 2),
            pv_truth - delta,
            pv_truth + delta,
            color="green",
            alpha=0.2,
            label="resolution window",
        )
        plt.ylabel("frequncy", fontsize=fontsize)
        plt.xlabel("z0 [cm]", fontsize=fontsize)
        plt.legend(fontsize=fontsize)

    def fh_pv_association(
        self, distance_from_pv: np.array, eta: np.array
    ) -> np.array(np.float32):

        eta_bin = np.digitize(np.abs(eta), self.eta_bins)
        assoc = distance_from_pv < self.deltaz_bins[eta_bin]

        return np.array(assoc, dtype=np.float32)

    def calculate_weighted_mean(self, h: np.array, be: np.array) -> float:
        x_i = 0.5 * (be[1:] + be[:-1])
        N = np.sum(h)

        return np.sum(h * x_i) / N

    def fh(
        self,
        z0: np.array,
        pt: np.array,
        eta: np.array,
        bin_edges: np.array,
        weighted_mean: bool = False,
    ) -> pd.Series:
        "Runs fast histo on a single event"

        z0 = z0.values
        pt = pt.values
        ntracks = z0.shape[0]

        if self.rank_by_pt:

            idx = pt.argsort()[::-1]

            pt = pt[idx][: self.rank_limit_n]
            z0 = z0[idx][: self.rank_limit_n]

        histo = np.histogram(z0, bins=bin_edges, weights=pt)[0]

        histo = np.convolve(histo, [1, 1, 1], mode="same")
        max_index = np.argmax(histo)
        if weighted_mean:
            z0_pv = self.calculate_weighted_mean(histo, bin_edges)
        else:
            z0_pv = self.z0_reco_fh_bin(max_index)
        z0_array = z0_pv * np.ones(ntracks, dtype=np.float32)

        return pd.Series(z0_array)

    def run_fh(self, weighted_mean: bool = False):
        "Runs fast histo on all events"
        bin_edges = self.fh_bins
        pv_fh = self.trk.groupby(level=0).progress_apply(
            lambda x: self.fh(
                x["trk_z0"], x["trk_pt"], x["trk_eta"], bin_edges, weighted_mean
            )
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
        self.fh_complete = True

    def calculate_weighted_mean(self, h: np.array, be: np.array) -> float:
        x_i = 0.5 * (be[1:] + be[:-1])
        N = np.sum(h)

        return np.sum(h * x_i) / N

    def dbscan(
        self,
        z0: np.array,
        pt: pd.Series,
        eps: float = 0.08,
        minPts: int = 2,
        pt_weighted: bool = False,
        stat: str = "median",
        threshold: float = 0.2,
        skim: float = 0.2,
        iqr_x: list = [25, 75],
        nbins: int = 50,
        weighted_mean: bool = False,
        convolve: bool = True,
        no_pt: bool = False,
        is_3d: bool = False,
        eta: np.array=None,
        phi: np.array=None,
        norm:bool=False,
        z0_sc:np.array=None,
        eta_sc:np.array=None,
        phi_sc:np.array=None,
    ) -> pd.Series:

        _df = pd.DataFrame({})
        
        if is_3d:
            _df['z0'] = z0
            _df['eta'] = eta
            _df['phi'] = phi
            _df['pt'] = pt
            if norm:
                _df['z0_sc'] = z0_sc
                _df['eta_sc'] = eta_sc
                _df['phi_sc'] = phi_sc
            pt_idx = 3
        else:
            _df["z0"] = z0
            _df["pt"] = pt
            pt_idx = 1

        if pt_weighted == False:
            if self.rank_by_pt and not self.use_classifier_cut:
                idx = pt.values.argsort()[::-1]
                top_idx = idx[: self.rank_limit_n]
                if is_3d:
                    if norm:
                        feature_idx =[4,5,6]
                    else:
                        feature_idx = [0,1,2]
                    X = _df.iloc[top_idx, feature_idx].values
                    db_clustering = DBSCAN(eps=eps, min_samples=minPts).fit(X)
                else:
                    z0_t = _df.iloc[top_idx, 0].values
                    db_clustering = DBSCAN(eps=eps, min_samples=minPts).fit(
                        z0_t.reshape(-1, 1)
                    )
            elif self.use_classifier_cut and not self.rank_by_pt:
                pred_prob = self.clf.predict_proba(_df["pt"].values.reshape(-1, 1))
                _df["prob"] = pred_prob[:, 1]
                _df["mask"] = _df["prob"] >= self.classifier_threshold
                db_clustering = DBSCAN(eps=eps, min_samples=minPts).fit(
                    _df.loc[_df["mask"], "z0"].values.reshape(-1, 1)
                )

            else:
                if is_3d:
                    if norm:
                        features = ['z0_sc','eta_sc','phi_sc']
                    else:
                        features = ['z0','eta','phi']
                    db_clustering = DBSCAN(eps=eps, min_samples=minPts).fit(
                       _df[features].values)
                else:
                   db_clustering = DBSCAN(eps=eps, min_samples=minPts).fit(
                        _df["z0"].values.reshape(-1, 1)
                    )
        elif pt_weighted == True:
            db_clustering = DBSCAN(eps=eps, min_samples=minPts).fit(
                _df["z0"].values.reshape(-1, 1), sample_weight=_df["pt"].values
            )

        if self.rank_by_pt and not self.use_classifier_cut:
            _df["db_label"] = -1
            _df.iloc[top_idx, 2] = db_clustering.labels_
        elif self.use_classifier_cut and not self.rank_by_pt:
            _df["db_label"] = -1
            _df.loc[_df["mask"], "db_label"] = db_clustering.labels_

        else:
            _df["db_label"] = db_clustering.labels_

        # Negative labels correspond to noise points, so floor pt 0 so they don't become the PV
        _df.loc[_df["db_label"] < 0, "pt"] = 0

        # Determine which DBSCAN label corresponds to the primary vertex.
        pv_label = (
            _df.groupby(["db_label"])["pt"].sum().sort_values(ascending=False).index[0]
        )

        _df["db_pv_label"] = 0

        _df.loc[_df["db_label"] == pv_label, "db_pv_label"] = 1
        if stat == "median":
            z0_reco = np.median(_df.loc[_df["db_pv_label"] == 1, "z0"])
        elif stat == "mean":
            z0_reco = np.mean(_df.loc[_df["db_pv_label"] == 1, "z0"])
        elif stat == "skim_mean":
            # take the mean if the values fall within the 20%-80% values of the group

            min_value = np.min(_df.loc[_df["db_pv_label"] == 1, "z0"])
            max_value = np.max(_df.loc[_df["db_pv_label"] == 1, "z0"])

            d = skim * np.abs(max_value - min_value)
            if (_df["db_pv_label"].sum() > 3) & (d > threshold):
                max_value = max_value - d
                min_value = min_value + d

                mask_min = _df["z0"] >= min_value
                mask_max = _df["z0"] <= max_value

                z0_reco = np.mean(
                    _df.loc[(_df["db_pv_label"] == 1) & mask_min & mask_max, "z0"]
                )
            else:
                z0_reco = np.mean(_df.loc[(_df["db_pv_label"] == 1), "z0"])
        elif stat == "skim_iqr":
            if _df["db_pv_label"].sum() > 3:

                q25 = np.percentile(_df.loc[_df["db_pv_label"] == 1, "z0"], iqr_x[0])
                q75 = np.percentile(_df.loc[_df["db_pv_label"] == 1, "z0"], iqr_x[1])

                mask_min = _df["z0"] >= q25
                mask_max = _df["z0"] <= q75

                z0_reco = np.mean(
                    _df.loc[(_df["db_pv_label"] == 1) & mask_min & mask_max, "z0"]
                )

            else:
                z0_reco = np.mean(_df.loc[(_df["db_pv_label"] == 1), "z0"])

        elif stat == "fast_histo":
            z0_pv = _df.loc[_df["db_pv_label"] == 1, "z0"]
            z0_low = np.min(z0_pv)
            z0_high = np.max(z0_pv)
            pt_pv = _df.loc[_df["db_pv_label"] == 1, "pt"]
            bin_edges = np.linspace(z0_low, z0_high, nbins)
            if no_pt:
                histo = np.histogram(z0_pv, bins=bin_edges)[0]
            else:
                histo = np.histogram(z0_pv, bins=bin_edges, weights=pt_pv)[0]
            if convolve:
                histo = np.convolve(histo, [1, 1, 1], mode="same")

            if weighted_mean:
                z0_reco = self.calculate_weighted_mean(histo, bin_edges)
            else:
                max_index = np.argmax(histo)

                half_bin_width = 0.5 * (bin_edges[1] - bin_edges[0])
                lowest_bin = bin_edges[0]
                highest_bin = bin_edges[-1]
                z0_reco = (
                    lowest_bin
                    + (highest_bin - lowest_bin) * (max_index / nbins)
                    + half_bin_width
                )
        _df["z0_reco_db"] = z0_reco

        return _df[["db_pv_label", "z0_reco_db"]]

    def run_dbscan(
        self,
        eps: float = 0.08,
        minPts: int = 2,
        pt_weighted: bool = False,
        stat: str = "median",
        threshold: float = 0.2,
        skim: float = 0.2,
        iqr_x: list = [25, 75],
        nbins: int = 50,
        weighted_mean: bool = False,
        convolve: bool = True,
        no_pt: bool = False,
        is_3d: bool=False,
        norm:bool=False,
    ):
        if is_3d:
            if norm == True:
                pv_dbscan = self.trk.groupby(level=0).progress_apply(
                    lambda x: self.dbscan(
                        x["trk_z0"],
                        x["trk_pt"],
                        eps,
                        minPts,
                        pt_weighted,
                        stat,
                        threshold,
                        skim,
                        iqr_x,
                        nbins,
                        weighted_mean,
                        convolve,
                        no_pt,
                        is_3d,
                        x['trk_eta'],
                        x['trk_phi'],
                        norm,
                        x["trk_z0_sc"],
                        x["trk_eta_sc"],
                        x["trk_phi_sc"]
                    )
                )
            else:
                pv_dbscan = self.trk.groupby(level=0).progress_apply(
                    lambda x: self.dbscan(
                        x["trk_z0"],
                        x["trk_pt"],
                        eps,
                        minPts,
                        pt_weighted,
                        stat,
                        threshold,
                        skim,
                        iqr_x,
                        nbins,
                        weighted_mean,
                        convolve,
                        no_pt,
                        is_3d,
                        x['trk_eta'],
                        x['trk_phi']
                    )
                )
        else:                     
            pv_dbscan = self.trk.groupby(level=0).progress_apply(
                lambda x: self.dbscan(
                    x["trk_z0"],
                    x["trk_pt"],
                    eps,
                    minPts,
                    pt_weighted,
                    stat,
                    threshold,
                    skim,
                    iqr_x,
                    nbins,
                    weighted_mean,
                    convolve,
                    no_pt,
                )
            )

        self.trk["z0_reco_db"] = pv_dbscan["z0_reco_db"]
        self.z0_reco_db = self.trk.groupby(level=0)["z0_reco_db"].first().values

        self.trk["trk_pv_assoc_db"] = pv_dbscan["db_pv_label"]

        print(f"Ran DBSCAN with eps: {eps}, minPts: {minPts}")
        self.db_complete = True

    def run_batched_dbscan(
        self,
        batch_size: int = 50,
        eps: float = 0.15,
        max_number_of_tracks: int = 232,
        verbose: bool = False,
        save_intermediate: bool = False,
        fh_metric: bool = False,
        at_end: bool = False,
        rank_by_pt: bool = False,
        top_pt_n: int = 10,
        fh_nbins: int = 5,
    ):

        self.trk["z0_reco_bdb"] = 999
        self.trk["distance_from_pv_bdb"] = 999
        self.trk["trk_pv_assoc_bdb"] = 0

        for i in tqdm(range(self.nevents)):

            mask = self.trk["event_number"] == i
            V = self.trk.loc[mask, ["trk_z0", "trk_pt"]].values
            z0 = V[:, 0]
            pt = V[:, 1]

            _db = BatchedDBSCAN(
                z0,
                pt,
                eps,
                batch_size,
                max_number_of_tracks,
                verbose=verbose,
                save_intermediate=save_intermediate,
                fh_metric=fh_metric,
                at_end=at_end,
                rank_by_pt=rank_by_pt,
                top_pt_n=top_pt_n,
                fh_nbins=fh_nbins,
            )
            _db.fit()
            if fh_metric:
                z0_reco = _db.z0_pv_wm
            else:
                z0_reco = _db.z0_pv

            self.trk.loc[mask, "z0_reco_bdb"] = z0_reco
            self.trk.loc[mask, "distance_from_pv_bdb"] = np.abs(
                self.trk.loc[mask, "z0_reco_bdb"] - self.trk.loc[mask, "trk_z0"]
            )

            self.trk.loc[mask, "trk_pv_assoc_bdb"] = self.fh_pv_association(
                self.trk.loc[mask, "distance_from_pv_bdb"],
                self.trk.loc[mask, "trk_eta"],
            )

        self.bdb_classification_metrics = self.trk_vertex_association(
            self.trk["is_pv"].values, self.trk["trk_pv_assoc_bdb"].values
        )
        self.z0_reco_bdb = self.trk["z0_reco_bdb"].groupby("entry").first().values

        print("Ran Batched DBSCAN")
        self.bdb_complete = True

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
        self.fh_complete = object.fh_complete
        self.db_complete = object.db_complete
        self.bdb_complete = object.bdb_complete
        self.eta_bins = np.array([0.0, 0.7, 1.0, 1.2, 1.6, 2.0, 2.4])
        self.deltaz_bins = np.array([0.0, 0.4, 0.6, 0.76, 1.0, 1.7, 2.2, 0.0])

        if self.db_complete:
            self.z0_reco_db = object.z0_reco_db
        if self.bdb_complete:
            self.z0_reco_bdb = object.z0_reco_bdb
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
        self.trk["distance_from_pv_fh_cor"] = np.abs(
            self.trk["z0_reco_fh_cor"] - self.trk["trk_z0"]
        )
        self.trk["trk_pv_assoc_fh_cor"] = self.fh_pv_association(
            self.trk["distance_from_pv_fh_cor"],self.trk["trk_eta"]
        )

        if self.db_complete:
            self.correct_bias_db()
            self.trk["distance_from_pv_db_cor"] = np.abs(
                self.trk["z0_reco_db_cor"] - self.trk["trk_z0"]
            )
            self.trk["trk_pv_assoc_db_cor"] = self.fh_pv_association(
                self.trk["distance_from_pv_db_cor"],self.trk["trk_eta"]
            )
        if self.bdb_complete:
            self.correct_bias_bdb()
            self.trk["distance_from_pv_bdb_cor"] = np.abs(
                self.trk["z0_reco_bdb_cor"] - self.trk["trk_z0"]
            )
            self.trk["trk_pv_assoc_bdb_cor"] = self.fh_pv_association(
                self.trk["distance_from_pv_bdb_cor"],self.trk["trk_eta"]
            )

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

    def correct_bias_bdb(self):
        self.z0_reco_bdb_cor = self.z0_reco_bdb.copy()
        self.trk["z0_reco_bdb_cor"] = self.trk["z0_reco_bdb"]
        positive_mask = self.z0_gen > 0
        negative_mask = self.z0_gen < 0

        positive_bias = np.median(
            (self.z0_gen[positive_mask] - self.z0_reco_bdb[positive_mask])
        )

        print("positive bias", positive_bias)
        self.z0_reco_bdb_cor[positive_mask] = (
            self.z0_reco_bdb_cor[positive_mask] + positive_bias
        )
        df_pos_mask = self.trk["event_number"].isin(self.positive_index)

        self.trk.loc[df_pos_mask, "z0_reco_bdb_cor"] = (
            self.trk.loc[df_pos_mask, "z0_reco_bdb_cor"] + positive_bias
        )

        negative_bias = np.median(
            (self.z0_gen[negative_mask] - self.z0_reco_bdb[negative_mask])
        )
        print("negative bias", negative_bias)

        self.z0_reco_bdb_cor[negative_mask] = (
            self.z0_reco_bdb_cor[negative_mask] + negative_bias
        )

        df_neg_mask = self.trk["event_number"].isin(self.negative_index)
        self.trk.loc[df_neg_mask, "z0_reco_bdb_cor"] = (
            self.trk.loc[df_neg_mask, "z0_reco_bdb_cor"] + negative_bias
        )

    def fh_pv_association(
        self, distance_from_pv: np.array, eta: np.array
    ) -> np.array(np.float32):

        eta_bin = np.digitize(np.abs(eta), self.eta_bins)
        assoc = distance_from_pv < self.deltaz_bins[eta_bin]

        return np.array(assoc, dtype=np.float32)

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
            # + "RMS: "
            # + str(RMS)
            + "\nquartile width: "
            + str(q_w)
            # + "\nmedian = "
            # + str(median)
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
        figsize: tuple = (8, 4),
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

        plt.figure(figsize=figsize)
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

    def compare_efficiency_z0(self, obj1, obj2,labels:list=["",""], filename: str = None, figsize=(10,6)):

        fig, ax = plt.subplots(
            1,
            1,
            figsize=figsize
        )

        ax.errorbar(
            obj1["x"],
            100 * obj1["eff"],
            xerr=obj1["error_x"],
            yerr=[100 * obj1["lower_error"], 100 * obj1["upper_error"]],
            ls="none",
            label=labels[0],
            color="#9E37AD",
        )
        ax.errorbar(
            obj2["x"],
            100 * obj2["eff"],
            xerr=obj2["error_x"],
            yerr=[100 * obj2["lower_error"], 100 * obj2["upper_error"]],
            ls="none",
            label=labels[1],
            color="#12A863",
        )
        ax.legend()
        ax.set_ylim(0, 110)
        ax.set_xlabel(r"$z_{0}^{PV, Gen}$ [cm]")
        ax.set_ylabel("Efficiency [%]")
        ax.set_title(
            "Primary Vertex Efficiency Comparison ($\delta = 0.1 cm$)", fontsize=24
        )
        if filename is not None:
            plt.savefig(filename, dpi=600, bbox_inches="tight")

    def compare_resolution_histogram(
        self, obj1: dict, obj2: dict, labels:list=None, filename: str = None
    ):

        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        
        hep.histplot(
            obj1["hist"], obj1["bins"], label=obj1["label"], ax=ax, color="#9E37AD"
        )

        hep.histplot(
            obj2["hist"], obj2["bins"], label=obj2["label"], ax=ax, color="#12A863"
        )

        ax.set_yscale("log")
        ax.set_xlabel(r"$z_0^{PV, Gen} - z_0^{PV, Reco}$ [cm]")
        ax.set_ylabel("Entries")
        ax.legend()
        if filename is not None:
            plt.savefig(filename, dpi=600, bbox_inches="tight")

    def compare_resolution_z0(self, obj1, obj2, filename: str = None):

        fig, ax = plt.subplots(1, 2, figsize=(20, 10), sharex=True, sharey=True)
        ax[0].errorbar(
            obj1["x"],
            obj1["mean_res"],
            xerr=obj1["xerr"],
            yerr=obj1["yerr"],
            ls="none",
            label=obj1["label"],
            color="#9E37AD",
        )
        ax[1].errorbar(
            obj2["x"],
            obj2["mean_res"],
            xerr=obj2["xerr"],
            yerr=obj2["yerr"],
            ls="none",
            label=obj2["label"],
            color="#12A863",
        )
        plt.subplots_adjust(wspace=0.01)
        ax[0].set_xlabel(r"$z^{PV}_{gen}$ [cm]")
        ax[1].set_xlabel(r"$z^{PV}_{gen}$ [cm]")
        ax[0].set_ylabel(r"$|z^{PV}_{gen} - z^{PV}_{reco}|$ [cm]")
        ax[0].set_title(obj1["label"])
        ax[1].set_title(obj2["label"])
        if filename is not None:
            plt.savefig(filename, dpi=500, bbox_inches="tight")

    def make_confusion_matrix(
        self,
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


class BatchedDBSCAN:
    def __init__(
        self,
        z0,
        pt,
        eps,
        batch_size,
        max_number_of_tracks,
        verbose: bool = False,
        save_intermediate: bool = False,
        fh_metric: bool = False,
        at_end: bool = False,
        rank_by_pt: bool = False,
        top_pt_n: int = 10,
        fh_nbins: int = 5,
    ):

        # predefined constants
        self.z0_boundary = 21
        self.pt_boundary = 0
        self.minPts = 2

        self.eps = eps
        self.batch_size = batch_size
        self.max_number_of_tracks = int(max_number_of_tracks)
        self.n_batches = math.ceil(self.max_number_of_tracks / self.batch_size)
        self.fh_metric = fh_metric
        self.at_end = at_end
        self.rank_by_pt = rank_by_pt
        self.top_pt_n = top_pt_n
        self.fh_nbins = fh_nbins

        # Max number of tracks including all batches
        self.max_n_tracks_batched = self.batch_size * self.n_batches
        self.max_n_clusters_batch = math.ceil(self.batch_size / self.minPts)  #
        self.max_n_clusters = math.ceil(self.max_n_tracks_batched / self.minPts)

        # need to pad vectors to have the fixed size
        n_pad = self.max_number_of_tracks - z0.shape[0]
        self.z0 = self.pad_vector(z0, n_pad, self.z0_boundary)
        self.pt = self.pad_vector(pt, n_pad, self.pt_boundary)
        # self.oneOverR = np.array([convert_pt_to_oneOverR(x) if x!=0 else 0 for x in self.pt0])
        # self.pt = np.array([convert_oneOverR_to_pt(x) if x!=0 else 0 for x in self.oneOverR])

        # Prefix sum variables
        self.max_number_of_tracks_power_2 = (
            1 << (self.max_number_of_tracks - 1).bit_length()
        )
        self.batch_size_power_2 = 1 << (self.batch_size - 1).bit_length()
        self.max_number_of_tracks_log_2 = np.log2(self.max_number_of_tracks_power_2)
        self.batch_size_log_2 = np.log2(self.batch_size_power_2)

        # Dictionaries for recording the results
        self.results = {}
        self.results_sklearn = {}
        self.merged_list = []

        # Debug options
        self.verbose = verbose
        self.save_intermediate = save_intermediate

        # Cluster array indices
        # pT, z0_low, z0_high, Noise
        self.pt_idx = 0
        self.z0_low_idx = 1
        self.z0_high_idx = 2
        self.noise_idx = 3

    def print_state(self):

        print(f"Max number of tracks: {self.max_number_of_tracks}")
        print(f"batch size: {self.batch_size}")
        print(f"n_batches: {self.n_batches}")
        print(f"max number of tracks batched: {self.max_n_tracks_batched}")
        print(f"max number of clusters per batch: {self.max_n_clusters_batch}")
        print(f"max number of clusters: {self.max_n_clusters}")
        print(f"z0 shape: {self.z0.shape[0]}")
        print(
            f"max_number of tracks rounded to the nearest power of 2: {self.max_number_of_tracks_power_2}"
        )
        print(f"max number of tracks log base2: {self.max_number_of_tracks_log_2}")
        print(f"batch size to the nearest power of 2: {self.batch_size_power_2}")
        print(f"batch size to log base2: {self.batch_size_log_2}")

    def pad_vector(self, vec: np.array, n_pad: int, value: int):
        """pads vector to a set size with given value"""

        vec_to_pad = value * np.ones(n_pad)

        vec = np.append(vec, vec_to_pad)

        return vec

    def build_tracks(self, z0, pt):
        """builds tracks batches"""

        # Shape is determined by the size of the batch, z0, pt and label (not used atm)
        track_batch = np.zeros((z0.shape[0], 3))
        track_batch[:, 0] = z0
        track_batch[:, 1] = pt

        # sort the tracks by z0
        track_batch = track_batch[track_batch[:, 0].argsort()]

        return track_batch

    def prefix_sum(self, arr):
        """
        Calculates the prefix sum of pT.
        Warning, requires array to be of size thats log base of 2.
        """
        size_log2 = int(np.log2(arr.shape[0]))

        # up-sweep
        for d in range(0, size_log2, 1):
            step_size = 2**d
            double_step_size = step_size * 2

            for i in range(0, arr.shape[0], double_step_size):
                arr[i + double_step_size - 1] += arr[i + step_size - 1]

        # down-sweep
        arr[arr.shape[0] - 1] = 0
        d = size_log2 - 1

        while d >= 0:
            step_size = 2**d
            double_step_size = step_size * 2
            for i in range(0, arr.shape[0], double_step_size):
                tmp = arr[i + step_size - 1]
                arr[i + step_size - 1] = arr[i + double_step_size - 1]
                arr[i + double_step_size - 1] += tmp
            d -= 1

        return arr

    def find_left_boundaries(self, tracks):

        left_boundaries = np.zeros(tracks.shape[0], dtype=bool)

        # first value is always a left boundary
        left_boundaries[0] = 1

        for i in range(1, left_boundaries.shape[0]):
            _t = tracks[i]

            if _t[0] - tracks[i - 1][0] > self.eps:
                tracks[i][2] = -1
                left_boundaries[i] = 1
            else:
                left_boundaries[i] = 0

        self.left_boundaries = left_boundaries
        return left_boundaries

    def find_right_boundaries(self, left_boundaries, rs, tracks):

        # max_tracks = self.batch_size
        max_tracks = tracks.shape[0]

        boundaries = np.zeros((max_tracks, 7))
        is_noise = np.ones((max_tracks, 1))

        for i in range(max_tracks - 1):

            left_edge = left_boundaries[i] and not (left_boundaries[i + 1])  # 1, 0
            right_edge = not (left_boundaries[i]) and left_boundaries[i + 1]  # 0, 1
            check_noise = (left_boundaries[i] == 1) and (left_boundaries[i + 1] == 1)

            if left_edge or right_edge:
                boundaries[i][0] = i
                boundaries[i][1] = rs[i]
                boundaries[i][2] = rs[i + 1]
                boundaries[i][3] = rs[i + 1] - rs[i]
                boundaries[i][4] = tracks[i, 0]
                boundaries[i][5] = tracks[i + 1, 0]
                is_noise[i] = 0
            elif check_noise:
                boundaries[i][0] = i
                boundaries[i][1] = rs[i]
                boundaries[i][2] = rs[i + 1]
                boundaries[i][3] = rs[i + 1] - rs[i]
                boundaries[i][4] = tracks[i, 0]
                boundaries[i][5] = tracks[i, 0]
                boundaries[i][6] = 1
                # is_noise[i] = 1
            else:
                boundaries[i][0] = max_tracks
                boundaries[i][1] = 0
                boundaries[i][2] = 0
                boundaries[i][3] = 0
                boundaries[i][4] = 21
                boundaries[i][5] = 21
                is_noise[i] = 0

        # Check for the last boundary

        if left_boundaries[max_tracks - 1] and is_noise[max_tracks - 1] == 0:
            boundaries[max_tracks - 1][0] = max_tracks
            boundaries[max_tracks - 1][1] = 0
            boundaries[max_tracks - 1][2] = 0
            boundaries[max_tracks - 1][3] = 0
            boundaries[max_tracks - 1][4] = 21
            boundaries[max_tracks - 1][5] = 21
        elif left_boundaries[max_tracks - 1] and is_noise[max_tracks - 1] == 1:

            boundaries[max_tracks - 1][0] = max_tracks - 1
            boundaries[max_tracks - 1][1] = rs[max_tracks - 1]
            boundaries[max_tracks - 1][2] = rs[max_tracks]
            boundaries[max_tracks - 1][3] = rs[max_tracks] - rs[max_tracks - 1]
            boundaries[max_tracks - 1][4] = tracks[max_tracks - 1, 0]
            boundaries[max_tracks - 1][5] = tracks[max_tracks - 1, 0]
            boundaries[max_tracks - 1][6] = 1
        else:
            boundaries[max_tracks - 1][0] = max_tracks - 1
            boundaries[max_tracks - 1][1] = rs[max_tracks - 1]
            boundaries[max_tracks - 1][2] = rs[max_tracks]
            boundaries[max_tracks - 1][3] = rs[max_tracks] - rs[max_tracks - 1]
            boundaries[max_tracks - 1][4] = tracks[max_tracks - 1, 0]
            boundaries[max_tracks - 1][5] = tracks[max_tracks - 1, 0]

        # boundaries[:, 6] = is_noise
        # Sort boundaries by the index
        sort_idx = boundaries[:, 0].argsort()
        boundaries = boundaries[sort_idx]
        is_noise = is_noise[sort_idx]
        self.is_noise = is_noise
        return boundaries

    def get_vertex(self, cluster_of_tracks: np.array) -> float:
        """
        Calculates the median z0 of the cluster of tracks
        """

        n_size = cluster_of_tracks.shape[0]

        if n_size % 2 == 0:
            return 0.5 * (
                cluster_of_tracks[n_size // 2] + cluster_of_tracks[n_size // 2 - 1]
            )
        else:
            return cluster_of_tracks[n_size // 2]

    def initialize_clusters(self, max_n_clusters: int) -> np.array:

        clusters = np.zeros((max_n_clusters, 4))

        clusters[:, self.z0_low_idx] = 21
        clusters[:, self.z0_high_idx] = 21

        return clusters

    def convert_boundaries_to_clusters(self, boundaries: np.array):
        bound_i = 0
        cluster_j = 0
        n_boundaries = boundaries.shape[0]
        if self.rank_by_pt:
            clusters = self.initialize_clusters(self.top_pt_n)
        else:
            clusters = self.initialize_clusters(n_boundaries)

        for _ in range(n_boundaries):
            # if boundaries[bound_i, 4] == 21:
            #     break
            if (bound_i + 2) >= n_boundaries:
                # print("breakpoint")
                break
            noise = boundaries[bound_i, -1]
            # print("bound_i: ", bound_i, "cluster_j: ", cluster_j, "noise: ", noise)

            if noise:
                z0_low = boundaries[bound_i, 4]
                z0_high = boundaries[bound_i, 5]
                pt_sum = boundaries[bound_i, 2] - boundaries[bound_i, 1]

                clusters[cluster_j, self.z0_low_idx] = z0_low
                clusters[cluster_j, self.z0_high_idx] = z0_high
                clusters[cluster_j, self.pt_idx] = pt_sum
                clusters[cluster_j, self.noise_idx] = noise

                if self.fh_metric and not self.at_end:
                    x = z0_low  # Since its noise, the x is the same as z0_low or z0_high
                    h = pt_sum  # Since its noise, the histo is just the pt of the noise point
                    self.fxs[(self.batch_number * self.batch_size) + cluster_j] = x * h
                    self.Ns[(self.batch_number * self.batch_size) + cluster_j] = h

                bound_i += 1
                cluster_j += 1
            else:
                z0_low = boundaries[bound_i, 4]
                z0_high = boundaries[bound_i + 1, 4]
                pt_sum = boundaries[bound_i + 1, 2] - boundaries[bound_i, 1]

                clusters[cluster_j, self.z0_low_idx] = z0_low
                clusters[cluster_j, self.z0_high_idx] = z0_high
                clusters[cluster_j, self.pt_idx] = pt_sum
                clusters[cluster_j, self.noise_idx] = noise

                if self.fh_metric and not self.at_end:
                    idx = boundaries[bound_i, 0]
                    idx_next = boundaries[bound_i + 1, 0]
                    f_batch_mask = np.zeros(n_boundaries)
                    bin_edges = np.linspace(z0_low, z0_high, self.fh_nbins)
                    x = 0.5 * (bin_edges[1:] + bin_edges[:-1])
                    f_batch_mask[int(idx) : int(idx_next)] = 1
                    h = np.histogram(
                        self.z0_batches[self.batch_number],
                        bin_edges,
                        weights=self.pt_batches[self.batch_number] * f_batch_mask,
                    )[0]

                    if self.rank_by_pt:
                        self.fxs[
                            (self.batch_number * self.top_pt_n) + cluster_j
                        ] = np.sum(x * h)
                        self.Ns[
                            (self.batch_number * self.top_pt_n) + cluster_j
                        ] = np.sum(h)
                    else:
                        self.fxs[
                            (self.batch_number * self.batch_size) + cluster_j
                        ] = np.sum(x * h)
                        self.Ns[
                            (self.batch_number * self.batch_size) + cluster_j
                        ] = np.sum(h)
                bound_i += 2
                cluster_j += 1

            # if bound_i >= n_boundaries:
            #     print("break")
            #     break

        return clusters

    def clusters_overlap(self, ci, cj) -> bool:
        # |---- c_i -----|   |---- c_j ----|
        case1 = ci[self.z0_low_idx] - self.eps <= cj[self.z0_high_idx]
        case2 = ci[self.z0_high_idx] + self.eps >= cj[self.z0_low_idx]

        overlap = case1 and case2

        return overlap

    def record_merging(self, ci, cj, cn):

        zi_low = round(ci[self.z0_low_idx], 2)
        zi_high = round(ci[self.z0_high_idx], 2)
        zi_pt = round(ci[self.pt_idx], 2)
        ci_str = f"[{zi_low}, {zi_high}, {zi_pt}]"

        zj_low = round(cj[self.z0_low_idx], 2)
        zj_high = round(cj[self.z0_high_idx], 2)
        zj_pt = round(cj[self.pt_idx], 2)
        cj_str = f"[{zj_low}, {zj_high}, {zj_pt}]"

        zn_low = round(cn[self.z0_low_idx], 2)
        zn_high = round(cn[self.z0_high_idx], 2)
        zn_pt = round(cn[self.pt_idx], 2)
        cn_str = f"[{zn_low}, {zn_high}, {zn_pt}]"

        merged_str = ci_str + " + " + cj_str + " -> " + cn_str
        self.merged_list.append(merged_str)

    def merge_clusters(self, c: np.array) -> np.array:

        clusters = c.copy()

        n_clusters = clusters.shape[0]
        if self.n_batches == 1:
            self.max_pt_i = np.argmax(clusters[:, self.pt_idx])
            self.max_pt = clusters[self.max_pt_i, self.pt_idx]
            self.merge_count = 0
            return clusters

        else:
            max_pt = 0
            max_pt_i = 0
            merge_count = 0

            comb = list(itertools.combinations(range(n_clusters), 2))
            self.comb = comb

            to_merge = 9 * np.ones((n_clusters, n_clusters))

            for i, j in comb:

                # skip if cluster  is outside detector
                if (clusters[i, self.z0_low_idx] >= 21) or (
                    clusters[j, self.z0_low_idx] >= 21
                ):
                    continue

                ci = copy.copy(clusters[i, :])
                cj = copy.copy(clusters[j, :])

                overlap = self.clusters_overlap(clusters[i, :], clusters[j, :])
                to_merge[i, j] = overlap
                if overlap:

                    # If cluster j is noise, then upon merging it is no-longer noise
                    cj_noise = clusters[j, self.noise_idx]

                    if cj_noise:
                        clusters[j, self.noise_idx] = 0

                    merge_count += 1

                    # Expand boundaries of cluster after merging
                    if clusters[i, self.z0_low_idx] < clusters[j, self.z0_low_idx]:
                        clusters[j, self.z0_low_idx] = clusters[i, self.z0_low_idx]
                    if clusters[i, self.z0_high_idx] > clusters[j, self.z0_high_idx]:
                        clusters[j, self.z0_high_idx] = clusters[i, self.z0_high_idx]

                    # Add the pT of the cluster being merged.
                    clusters[j, self.pt_idx] += clusters[i, self.pt_idx]

                    if self.fh_metric and not self.at_end:
                        self.fxs[j] += self.fxs[i]
                        self.Ns[j] += self.Ns[i]

                    # Erase merged cluster.
                    clusters[i, self.pt_idx] = 0
                    clusters[i, self.z0_low_idx] = 21
                    clusters[i, self.z0_high_idx] = 21
                    clusters[i, self.noise_idx] = 0

                    self.record_merging(ci, cj, clusters[j, :])

                # check if the pT_sum max is now higher
                # Need to protect against selecting a noise point as PV
                if (max_pt < clusters[j, self.pt_idx]) and (
                    clusters[j, self.noise_idx] != 1
                ):
                    max_pt = clusters[j, self.pt_idx]
                    max_pt_i = j
            self.to_merge = pd.DataFrame(to_merge)
            self.max_pt = max_pt
            self.max_pt_i = max_pt_i
            self.merge_count = merge_count

            if self.fh_metric and not self.at_end:

                self.z0_pv_wm = self.fxs[max_pt_i] / self.Ns[max_pt_i]

            return pd.DataFrame(
                clusters, columns=["pt_sum", "z0_low", "z0_high", "noise"]
            )

    def fitsklearn(self):
        start_idx = 0
        end_idx = start_idx + self.batch_size
        n_pad = (self.n_batches * self.batch_size) - self.z0.shape[0]
        self.z0 = self.pad_vector(self.z0, n_pad, 21)
        self.pt = self.pad_vector(self.pt, n_pad, 0)

        clusters_df = pd.DataFrame({})

        clusters = self.initialize_clusters(self.max_n_tracks_batched)

        for i in range(self.n_batches):
            start_idx = i * self.batch_size
            end_idx = (i + 1) * self.batch_size
            z0_batch = self.z0[start_idx:end_idx]
            pt_batch = self.pt[start_idx:end_idx]

            _db = DBSCAN(eps=0.15, min_samples=2).fit(z0_batch.reshape(-1, 1))

            _results = pd.DataFrame(
                {"z0": z0_batch, "pt": pt_batch, "label": _db.labels_}
            )
            max_label = _results.label.max()
            n_noise = _results[_results.label == -1].shape[0]

            _results.loc[_results.label == -1, "label"] = (
                np.arange(n_noise) + max_label + 1
            )

            clusters_batch = _results.groupby(["label"]).agg(
                {"z0": [np.min, np.max], "pt": [np.sum, "count"]}
            )
            clusters_batch.columns = ["z0_low", "z0_high", "pt_sum", "ntracks"]
            clusters_batch["noise"] = 0
            clusters_batch.loc[clusters_batch["ntracks"] < 2, "noise"] = 1
            clusters_batch = clusters_batch.drop(columns=["ntracks"])

            if self.fh_metric and not self.at_end:
                fxs = []
                Ns = []
                for j in range(clusters_batch.shape[0]):
                    z0_low = clusters_batch["z0_low"][j]
                    z0_high = clusters_batch["z0_high"][j]
                    z0_cluster_tracks = z0_batch[
                        (z0_batch >= z0_low) & (z0_batch <= z0_high)
                    ]
                    pt_cluster_tracks = pt_batch[
                        (z0_batch >= z0_low) & (z0_batch <= z0_high)
                    ]
                    be = np.linspace(z0_low, z0_high, self.fh_nbins)
                    x = 0.5 * (be[1:] + be[:-1])
                    h, _ = np.histogram(
                        z0_cluster_tracks, bins=be, weights=pt_cluster_tracks
                    )
                    fx = np.sum(x * h)
                    fxs.append(fx)
                    N = np.sum(h)
                    Ns.append(N)

                clusters_batch["fx"] = fxs
                clusters_batch["N"] = Ns

            self.results_sklearn[i] = clusters_batch

            clusters_df = pd.concat([clusters_df, clusters_batch])

        n_clusters = clusters_df.shape[0]

        clusters[0:n_clusters, self.pt_idx] = clusters_df["pt_sum"]
        clusters[0:n_clusters, self.z0_low_idx] = clusters_df["z0_low"]
        clusters[0:n_clusters, self.z0_high_idx] = clusters_df["z0_high"]
        clusters[0:n_clusters, self.noise_idx] = clusters_df["noise"]

        if self.fh_metric and not self.at_end:
            self.fxs = clusters_df["fx"].copy().values
            self.Ns = clusters_df["N"].copy().values

        self.clusters_unmerged = pd.DataFrame(
            clusters.copy(), columns=["pt_sum", "z0_low", "z0_high", "noise"]
        )

        clusters_merged = self.merge_clusters(clusters)
        self.clusters_merged = pd.DataFrame(
            clusters_merged, columns=["pt_sum", "z0_low", "z0_high", "noise"]
        )

        pv_z0_low = self.clusters_merged.iloc[self.max_pt_i, self.z0_low_idx]
        pv_z0_high = self.clusters_merged.iloc[self.max_pt_i, self.z0_high_idx]

        z0_pv = np.median(self.z0[(self.z0 >= pv_z0_low) & (self.z0 <= pv_z0_high)])
        if self.fh_metric and not self.at_end:
            self.z0_pv_skl = self.z0_pv_wm
        elif self.fh_metric and self.at_end:
            bin_edges = np.linspace(pv_z0_low, pv_z0_high, self.fh_nbins)
            x = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            h = np.histogram(
                self.z0[(self.z0 >= pv_z0_low) & (self.z0 <= pv_z0_high)],
                bin_edges,
                weights=self.pt[(self.z0 >= pv_z0_low) & (self.z0 <= pv_z0_high)],
            )[0]
            z0_pv = np.sum(x * h) / np.sum(h)
            self.z0_pv_wm = z0_pv
            self.z0_pv_skl = z0_pv
        else:
            self.z0_pv_skl = z0_pv

    def fit(self):

        np.set_printoptions(precision=2)
        np.set_printoptions(suppress=True)

        if self.verbose:
            self.print_state()

        start_idx = 0
        end_idx = start_idx + self.batch_size
        # Need to pad vectors to match the size of n_batches*batch_size
        n_pad = (self.n_batches * self.batch_size) - self.z0.shape[0]
        self.z0 = self.pad_vector(self.z0, n_pad, 21)
        self.pt = self.pad_vector(self.pt, n_pad, 0)

        if self.rank_by_pt:
            clusters = self.initialize_clusters(self.top_pt_n * self.n_batches)
        else:
            clusters = self.initialize_clusters(self.max_n_tracks_batched)

        self.z0_batches = {}
        self.pt_batches = {}
        self.rs_batches = {}
        self.left_boundaries_batches = {}
        self.boundaries_batches = {}
        self.clusters_batches = {}
        if self.fh_metric and not self.at_end:
            self.fxs = np.zeros(self.max_n_tracks_batched)
            self.Ns = np.zeros(self.max_n_tracks_batched)

        pv_cluster = np.zeros((1, 4))
        merge_count = 0
        for i in range(self.n_batches):
            # print(f"---------- batch number: {i} ---------")
            start_idx = i * self.batch_size
            end_idx = (i + 1) * self.batch_size
            self.batch_number = i

            # if i == 1:
            #     break

            z0_batch = self.z0[start_idx:end_idx]
            pt_batch = self.pt[start_idx:end_idx]

            if self.rank_by_pt:
                idx_sort = pt_batch.argsort()[::-1]
                z0_batch = z0_batch[idx_sort]
                pt_batch = pt_batch[idx_sort]
                z0_batch = z0_batch[: self.top_pt_n]
                pt_batch = pt_batch[: self.top_pt_n]

            track_batch = self.build_tracks(z0_batch, pt_batch)
            self.tracks = track_batch

            rs_batch = self.pad_vector(
                track_batch[:, 1], self.batch_size_power_2 - track_batch.shape[0], 0
            )

            rs_batch = self.prefix_sum(rs_batch)
            self.rs = rs_batch

            # Storing batches
            self.z0_batches[i] = track_batch[:, 0]
            self.pt_batches[i] = track_batch[:, 1]
            self.rs_batches[i] = rs_batch

            # Finding Left Boundaries
            left_boundaries = self.find_left_boundaries(track_batch)
            self.left_boundaries_batches[i] = left_boundaries
            # print("left_boundaries shape: ", left_boundaries.shape)

            # Finding Right Boundaries
            np.save("left_boundaries.npy", left_boundaries)
            np.save("rs_batch.npy", rs_batch)
            np.save("track_batch.npy", track_batch)

            boundaries = self.find_right_boundaries(
                left_boundaries, rs_batch, track_batch
            )
            self.boundaries_batches[i] = boundaries
            # print("boundaries shape: ", boundaries.shape)

            self.boundaries = boundaries
            # np.save('boundaries.npy', boundaries)
            clusters_batch = self.convert_boundaries_to_clusters(boundaries)
            self.clusters_batches[i] = clusters_batch

            if self.rank_by_pt:
                clusters[
                    i * self.top_pt_n : (i + 1) * self.top_pt_n, :
                ] = clusters_batch
            else:
                clusters[
                    i * self.batch_size : (i + 1) * self.batch_size, :
                ] = clusters_batch

            self.results[i] = clusters_batch
        clusters = self.merge_clusters(clusters)

        self.clusters = clusters

        # Find pv_cluster
        # print(type(clusters))
        if self.n_batches == 1:
            pv_cluster[0, :] = clusters[self.max_pt_i, :]
        else:
            pv_cluster[0, :] = clusters.iloc[self.max_pt_i, :]

        # print(self.max_pt, self.max_pt_i)
        # print(f"Merged count: {self.merge_count}")

        pv_tracks = []
        pv_pt = []

        for i in range(self.max_number_of_tracks):
            z0_trk = self.z0[i]

            if (z0_trk >= pv_cluster[0, self.z0_low_idx]) and (
                z0_trk <= pv_cluster[0, self.z0_high_idx]
            ):
                pv_tracks.append(z0_trk)
                pv_pt.append(self.pt[i])

        if self.fh_metric and self.at_end:
            bin_edges = np.linspace(
                pv_cluster[0, self.z0_low_idx],
                pv_cluster[0, self.z0_high_idx],
                self.fh_nbins,
            )
            x = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            h = np.histogram(pv_tracks, bin_edges, weights=pv_pt)[0]

            z0_pv = np.sum(h * x) / np.sum(h)
            self.z0_pv = z0_pv
            self.z0_pv_wm = z0_pv

        else:
            median_vertex = self.get_vertex(np.array(pv_tracks))
            self.z0_pv = np.median(pv_tracks)

        # print(f"mean: {np.mean(pv_tracks)}")
        # print(f"median: {np.median(pv_tracks)}")
        # print(f"median2: {median_vertex}")

        
def generate_roc_curve(bkg:np.array, sig:np.array,mincut:float=None, maxcut:float=None, step:float = 0.1) -> tuple: 
    
    if maxcut is None:
        if np.max(bkg) > np.max(sig):
            maxcut = np.max(bkg)
        else:
            maxcut = np.max(sig)
     
    if mincut is None:
        if np.min(bkg) < np.min(sig):
            mincut = np.min(bkg)
        else:
            mincut = np.min(sig)
    
    print(mincut, maxcut, step)
    iterationset = np.arange(mincut, maxcut, step)
    
    tpr = []
    fpr = []
    
    for cut in iterationset:
        bkg_cut_less = np.where(bkg < cut)
        sig_cut_less = np.where(sig < cut)
        bkg_cut_more = np.where(bkg > cut)
        sig_cut_more = np.where(sig > cut)
        
        TP = float(sig_cut_less[0].shape[0])
        FN = float(sig_cut_more[0].shape[0])
        
        TN = float(bkg_cut_more[0].shape[0])
        FP = float(bkg_cut_less[0].shape[0])
        
        sensitivity = float(TP / (TP +FN))
        specitivity = float(TN / (TN + FP))
        one_min_spec = float(1-specitivity)
        print(sensitivity, one_min_spec)
        tpr.append(sensitivity)
        fpr.append(one_min_spec)
        
    roc_auc = auc(fpr, tpr)
    
    return roc_auc, fpr, tpr

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

def test_feature_discrimination(X:np.array,y:np.array,model:str="LR")-> float:

       
    x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.5, stratify=y)
    nsig = y_train[y_train==1].shape[0]
    nbkg = y_train[y_train==0].shape[0]
    R = nbkg/nsig
    if model == "LR":
        clf = LogisticRegression(class_weight='balanced').fit(x_train, y_train)
    elif model == "RF":
        clf = RandomForestClassifier().fit(x_train, y_train)
    elif model == "xgb":
        clf = XGBClassifier(scale_pos_weight=R).fit(x_train, y_train)
        
    y_pred_prob = clf.predict_proba(x_test)[:,1]
    
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    
    AUC = auc(fpr, tpr)
    
    return AUC, clf

from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

def test_xgb_clf(trk:pd.DataFrame, features:list, label_name:str, search_space:dict=None):
    
    if search_space is None:
        search_space = {
            "max_depth":[6,14],
            "learning_rate":[0.05, 0.1],
            "n_estimators":[50, 200],
        }
    X = trk[features].values
    y = trk[label_name].values
    
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scale_pos_weight = y_train[y_train==0].shape[0] / y_train[y_train==1].shape[0]
    
    xgb = XGBClassifier(seed=20, scale_pos_weight=scale_pos_weight)
        
        
    clf = GridSearchCV(
        estimator = xgb,
        param_grid=search_space,
        scoring="accuracy",
        verbose=3,
        cv=3
    )
    clf.fit(x_train, y_train)
    
    y_pred = clf.predict_proba(x_test)[:,1]
    
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    AUC = auc(fpr,tpr)
    
    return AUC, clf
        
        
    
        
    
    