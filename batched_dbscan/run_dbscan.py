import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from tqdm import tqdm

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

    return z0_pv, pt_pv


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
    z0_batched_skl_wm = []
    z0_batched_acc_wm = []
    z0_batched_acc_pt = []
    z0_batched_acc_pt_wm = []

    z0_mc = []

    pt_pvs = []
    pt_batched = []
    pt_batched_skl = []
    pt_batched_skl_wm = []
    pt_batched_acc_wm = []
    pt_batched_acc_pt = []
    pt_batched_acc_pt_wm = []

    mc_df = pd.read_pickle("/home/raichu/data/mc_25k.pkl")
    trk_df = pd.read_pickle("/home/raichu/data/trk_processed_25k.pkl")

    for i in tqdm(range(5000)):
        file_i = i

        # z0_file = f"/media/lucas/QS/binaries-trk/OldKF_TTbar_170K_quality-{file_i}-trk-z0.bin"
        # pt_file = f"/media/lucas/QS/binaries-trk/OldKF_TTbar_170K_quality-{file_i}-trk-pt.bin"
        store = "/home/raichu/data/data/binaries-trk-100/"
        # z0_file = store + f"b-{file_i}-trk-z0.bin"
        # pt_file = store + f"b-{file_i}-trk-pt.bin"
        # mc_file = store + f"b-{file_i}-trk-mc.bin"
        # z0 = np.fromfile(z0_file, dtype=np.float32)
        # pt = np.fromfile(pt_file, dtype=np.float32)
        # mc = np.fromfile(mc_file, dtype=np.float32)

        mc = mc_df.query(f"entry=={i}")["pv_MC"].values
        z0 = trk_df.query(f"entry=={i}")["trk_z0"].values
        pt = trk_df.query(f"entry=={i}")["trk_pt"].values
        # print(z0.shape[0])
        if z0.shape[0] >= 232:
            print("continued")
            continue

        z0_mc.append(mc)

        z0_pv, pt_pv = run_normal_dbscan(z0, pt, eps)

        #         db = BatchedDBSCAN(
        #             z0, pt, eps, batch_size, max_number_of_tracks, verbose, save_intermediate
        #         )

        #         db.fit()

        #         db_skl = BatchedDBSCAN(
        #             z0, pt, eps, batch_size, max_number_of_tracks, verbose, save_intermediate
        #         )

        #         db_skl.fitsklearn()

        #         db_fh = BatchedDBSCAN(
        #             z0, pt, eps, batch_size, max_number_of_tracks, verbose, save_intermediate, fh_metric=True, at_end=True)

        #         db_fh.fitsklearn()

        db_fh_acc = BatchedDBSCAN(
            z0,
            pt,
            eps,
            batch_size,
            max_number_of_tracks,
            verbose,
            save_intermediate,
            fh_metric=True,
            at_end=False,
        )

        db_fh_acc.fit()

        db_pt_wm = BatchedDBSCAN(
            z0,
            pt,
            eps,
            batch_size,
            max_number_of_tracks,
            verbose,
            save_intermediate,
            fh_metric=True,
            at_end=False,
            rank_by_pt=True,
            top_pt_n=10,
            fh_nbins=3,
        )

        db_pt_wm.fit()

        db_ptx = BatchedDBSCAN(
            z0,
            pt,
            eps,
            batch_size,
            max_number_of_tracks,
            verbose,
            save_intermediate,
            fh_metric=False,
            at_end=False,
            rank_by_pt=True,
            top_pt_n=10,
        )

        db_ptx.fit()

        z0_pvs.append(z0_pv)
        # z0_batched.append(db.z0_pv)
        # z0_batched_skl.append(db_skl.z0_pv_skl)
        # z0_batched_skl_wm.append(db_fh.z0_pv_skl)
        z0_batched_acc_wm.append(db_fh_acc.z0_pv_wm)
        z0_batched_acc_pt_wm.append(db_pt_wm.z0_pv_wm)
        z0_batched_acc_pt.append(db_ptx.z0_pv)

        pt_pvs.append(pt_pv)
        # pt_batched.append(db.max_pt)
        # pt_batched_skl.append(db_skl.max_pt)
        # pt_batched_skl_wm.append(db_fh.max_pt)
        pt_batched_acc_wm.append(db_fh_acc.max_pt)
        pt_batched_acc_pt_wm.append(db_pt_wm.max_pt)
        pt_batched_acc_pt.append(db_ptx.max_pt)

    r = pd.DataFrame(
        {
            "z0_mc": z0_mc,
            "z0_normal": z0_pvs,
            # "z0_batched": z0_batched,
            # "z0_batched_skl": z0_batched_skl,
            # "z0_batched_skl_wm": z0_batched_skl_wm,
            "z0_batched_acc_wm": z0_batched_acc_wm,
            "z0_batched_acc_pt_wm": z0_batched_acc_pt_wm,
            "z0_batched_acc_pt": z0_batched_acc_pt,
            "pt_normal": pt_pvs,
            # "pt_batched": pt_batched,
            # "pt_batched_skl": pt_batched_skl,
            # "pt_batched_skl_wm": pt_batched_skl_wm,
            "pt_batched_acc_wm": pt_batched_acc_wm,
            "pt_batched_acc_pt_wm": pt_batched_acc_pt_wm,
            "pt_batched_acc_pt": pt_batched_acc_pt,
        }
    )
    d = pd.DataFrame({})

    # d['z0_diff'] = 100 * (r['z0_batched'] - r['z0_normal']) / r['z0_normal']
    # d['pt_diff'] = 100 * (r['pt_batched'] - r['pt_normal']) / r['pt_normal']
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    print(r)

    for col in [
        "z0_normal",
        "z0_batched_acc_wm",
        "z0_batched_acc_pt_wm",
        "z0_batched_acc_pt",
    ]:

        eff = 100 * np.sum((np.abs(r["z0_mc"] - r[col]) < 0.15)) / r.shape[0]
        print(f"Efficiency {col}: ", eff)

    # print(d)

    # print(d.describe())
    # print(
    #     f"file {i}: {db.z0_pv} ({db_skl.z0_pv_skl}), {db.max_pt} ({db_skl.max_pt})"
    # )
    # print(db.boundaries_batches[0])
    # print(db.z0_pv, db.max_pt)

    # import json
    # with open('merged_list.json', 'w') as f:
    #     json.dump(db.merged_list, f, indent=4)
