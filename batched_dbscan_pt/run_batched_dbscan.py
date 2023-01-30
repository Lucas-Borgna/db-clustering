import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from tqdm import tqdm
from batched_dbscan import BatchedDBSCAN

np.random.seed(42)

pd.set_option("display.precision", 2)


def run_normal_dbscan(z0: np.array, pt: np.array, eps: float = 0.15) -> tuple:

    """Function runs the normal DBSCAN algorithm based on the sklearn package.
    It requires the inpu tracks z0 and pt values and the eps parameter.

    Args:
        z0 (np.array): array containing the z0 values of the tracks
        pt (np.array): array containing the pt values of the tracks
        eps (float): float value for the eps parameter of the DBSCAN algorithm

    Returns:
        (tuple): returns the z0 location of the primary vertex and its corresponding pt sum value
    """
    dfb = pd.DataFrame({"z0": z0, "pt": pt})
    db = DBSCAN(eps=eps, min_samples=2).fit(dfb["z0"].values.reshape(-1, 1))
    dfb["label"] = db.labels_

    # Sets all tracks identified as noise to have a pt value of 0 so they
    # are not considered in the primary vertex calculation
    dfb.loc[dfb.label == -1, "pt"] = 0

    # Groups the tracks into clusters and calculates the median z0 value and the sum of the pt values
    clusters = dfb.groupby("label").agg({"z0": [np.median], "pt": [np.sum]})
    clusters.columns = ["z0", "pt_sum"]

    # Highest pt sum cluster is the primary vertex
    clusters = clusters.sort_values(by="pt_sum", ascending=False)

    z0_pv = clusters.iloc[0]["z0"]
    pt_pv = clusters.iloc[0]["pt_sum"]

    return z0_pv, pt_pv


if __name__ == "__main__":
    """Program iterates through 100 events in the binary files and runs the normal and batched DBSCAN algorithms
    using the same data and compares it to the truth monte carlo data.
    Because the batched DBSCAN algorithm removes low pT tracks, we cannot use to compare the pT with the regular algorithm.
    """

    # Global configuration parameters
    max_number_of_tracks = 232
    max_number_of_tracks_power_2 = 256
    max_number_of_tracks_log_2 = 8
    batch_size = 50
    top_pt_n = 10
    fh_nbins = 5
    eps = 0.15
    # Location of the binary files
    store = "/home/lucas/Documents/RA/raichu/data/data/binaries-trk-100/"

    z0_mc = []
    z0_db_pv = []
    z0_bdb_pv = []

    for i in tqdm(range(100)):
        file_i = i

        z0_file = store + f"b-{file_i}-trk-z0.bin"
        pt_file = store + f"b-{file_i}-trk-pt.bin"
        mc_file = store + f"b-{file_i}-trk-mc.bin"
        z0 = np.fromfile(z0_file, dtype=np.float32)
        pt = np.fromfile(pt_file, dtype=np.float32)
        # Truth monte carlo data
        mc = np.fromfile(mc_file, dtype=np.float32)

        if z0.shape[0] >= max_number_of_tracks:
            print("continued")
            continue

        z0_mc.append(mc)

        z0_pv, _ = run_normal_dbscan(z0, pt, eps)
        z0_db_pv.append(z0_pv)

        # Runs the batched DBSCAN algorithm
        # Uses a weighted mean statistic by creating a histogram with fh_nbins
        bdb = BatchedDBSCAN(
            z0,
            pt,
            eps,
            batch_size,
            max_number_of_tracks,
            top_pt_n=top_pt_n,
            fh_nbins=fh_nbins,
        )

        bdb.fit()

        z0_bdb_pv.append(bdb.z0_pv_wm)

    r = pd.DataFrame(
        {
            "z0_mc": z0_mc,
            "z0_db": z0_db_pv,
            "z0_bdb": z0_bdb_pv,
        }
    )
    r["db_diff"] = np.abs(r["z0_mc"] - r["z0_db"])
    r["bdb_diff"] = np.abs(r["z0_mc"] - r["z0_bdb"])

    # An event is reconstructed correctly if it is within 0.15 cm of the truth monte carlo values
    db_eff = 100 * np.sum((r["db_diff"] < 0.15)) / r.shape[0]
    bdb_eff = 100 * np.sum((r["bdb_diff"] < 0.15)) / r.shape[0]

    print(f"Efficiency of regular DBSCAN: {db_eff}")
    print(f"Efficiency of batched DBSCAN (with pt selection): {bdb_eff}")
