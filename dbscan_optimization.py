import pandas as pd
import numpy as np
import glob
from scipy.sparse import data
from sklearn.cluster import DBSCAN

from notebooks.pv_utils import truth_pv_z0, run_dbscan, data_load


def run_optimization():

    storage_path = "/mnt/storage/lborgna/track/l1_nnt/"
    # input_tp_files = glob.glob(storage_path + "tp_??.pkl")
    input_files = [storage_path + "OldKF_TTbar_170K_quality.root"]
    # list_of_df = []

    minPts_values = [4]

    eps_values = [0.05, 0.06, 0.08, 0.1, 0.15, 0.20, 0.25]

    for i, input_file in enumerate(input_files):
        _results = pd.DataFrame({})

        # tp, trk = data_load(input_file)
        tp = pd.read_pickle("/mnt/storage/lborgna/track/l1_nnt/tp.pkl")
        trk = pd.read_pickle("/mnt/storage/lborgna/track/l1_nnt/trk.pkl")
        mc = pd.read_pickle("/mnt/storage/lborgna/track/l1_nnt/mc.pkl")

        z0_gen = truth_pv_z0(tp)
        _results["z0_gen"] = z0_gen
        _results["z0_MC"] = mc.reset_index()["pv_MC"]

        for minPts in minPts_values:
            for eps in eps_values:
                print("minPts: ", minPts, "\teps: ", eps)

                z0_dbscan = run_dbscan(
                    trk,
                    z0_column=["trk_z0"],
                    pt_column=["trk_pt"],
                    eps=eps,
                    minPts=minPts,
                    remove_noise=True,
                )

                _results[f"z0_dbscan_e{eps}_min_{minPts}"] = z0_dbscan["z0"]

        # list_of_df.append(_results)

    # optimization_results = pd.concat(list_of_df, ignore_index=True)

    _results.to_pickle("dbscan_optimization_new_file_remove_noise_min4.pkl")


if __name__ == "__main__":

    run_optimization()
