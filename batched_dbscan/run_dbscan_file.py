import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

np.random.seed(42)

pd.set_option("display.precision", 2)


def convert_pt_to_oneOverR(pt):

    return 0.3 * 3.811 / (100 * pt)


def convert_oneOverR_to_pt(oneOverR):

    return 0.3 * 3.811 / (100 * oneOverR)


def run_normal_dbscan(z0, pt, eps):
    dfb = pd.DataFrame({"z0": z0, "pt": pt})
    db = DBSCAN(eps=eps, min_samples=2).fit(dfb["z0"].values.reshape(-1, 1))
    dfb["label"] = db.labels_

    dfb.loc[dfb.label == -1, "pt"] = 0

    clusters = dfb.groupby("label").agg({"z0": [np.median], "pt": [np.sum]})
    clusters.columns = ["z0", "pt_sum"]

    clusters = clusters.sort_values(by="pt_sum", ascending=False)

    z0_pv = clusters.iloc[0]["z0"]
    pt_pv = clusters.iloc[0]["pt_sum"]

    return z0_pv, pt_pv, clusters


if __name__ == "__main__":

    max_number_of_tracks = 232
    max_number_of_tracks_power_2 = 256
    max_number_of_tracks_log_2 = 8
    batch_size = 50

    eps = 0.15
    verbose = False
    save_intermediate = False
    from batched_dbscan import BatchedDBSCAN

    z0_pvs = []
    z0_batched = []
    z0_batched_skl = []
    pt_pvs = []
    pt_batched = []
    pt_batched_skl = []
    file_i = 51

    store = "/home/kirby/data/binaries-trk-100/"
    z0_file = store + f"b-{file_i}-trk-z0.bin"
    pt_file = store + f"b-{file_i}-trk-pt.bin"
    z0 = np.fromfile(z0_file, dtype=np.float32)
    pt = np.fromfile(pt_file, dtype=np.float32)

    z0_pv, pt_pv, clusters = run_normal_dbscan(z0, pt, eps)

    db = BatchedDBSCAN(
        z0, pt, eps, batch_size, max_number_of_tracks, verbose, save_intermediate
    )

    db.fit()

    db_skl = BatchedDBSCAN(
        z0, pt, eps, batch_size, max_number_of_tracks, verbose, save_intermediate
    )

    db_skl.fitsklearn()

    z0_pvs.append(z0_pv)
    z0_batched.append(db.z0_pv)
    z0_batched_skl.append(db_skl.z0_pv_skl)
    pt_pvs.append(pt_pv)
    pt_batched.append(db.max_pt)
    pt_batched_skl.append(db_skl.max_pt)

    r = pd.DataFrame(
        {
            "z0_normal": z0_pvs,
            "z0_batched": z0_batched,
            "z0_batched_skl": z0_batched_skl,
            "pt_normal": pt_pvs,
            "pt_batched": pt_batched,
            "pt_batched_skl": pt_batched_skl,
        }
    )
    print(r)
    d = pd.DataFrame({})


    d['z0_diff'] = 100 * (r['z0_batched'] - r['z0_normal']) / r['z0_normal']
    d['pt_diff'] = 100 * (r['pt_batched'] - r['pt_normal']) / r['pt_normal']
    pd.set_option('display.max_columns', None)
    print(d)
    print(clusters)

    # print(d.describe())
    # print(
    #     f"file {i}: {db.z0_pv} ({db_skl.z0_pv_skl}), {db.max_pt} ({db_skl.max_pt})"
    # )
    # print(db.boundaries_batches[0])
    # print(db.z0_pv, db.max_pt)

    # import json
    # with open('merged_list.json', 'w') as f:
    #     json.dump(db.merged_list, f, indent=4)
