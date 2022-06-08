import pandas as pd
import numpy as np
import glob

from notebooks.pv_utils import run_fast_histo, data_load, truth_pv_z0


def run_optimization():

    bin_widths = [0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35]

    _results = pd.DataFrame({})

    tp = pd.read_pickle("/mnt/storage/lborgna/track/l1_nnt/tp.pkl")
    trk = pd.read_pickle("/mnt/storage/lborgna/track/l1_nnt/trk.pkl")
    mc = pd.read_pickle("/mnt/storage/lborgna/track/l1_nnt/mc.pkl")

    z0_gen = truth_pv_z0(tp)
    _results["z0_gen"] = z0_gen
    _results["z0_MC"] = mc.reset_index()["pv_MC"]

    for bin_width in bin_widths:
        print("bw: ", bin_width)
        be = np.arange(-15, 15 + bin_width, bin_width)

        z0_fastHisto = run_fast_histo(
            trk, bin_edges=be, z0_column="trk_z0", pt_column="trk_pt"
        )

        _results[4] = z0_fastHisto

    _results.to_pickle("fastHisto_optimization_new_file.pkl")


if __name__ == "__main__":

    run_optimization()
