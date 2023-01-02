import numpy as np
import math
import pandas as pd
import itertools
import copy

np.random.seed(42)
np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

if __name__ == "__main__":

    from batched_dbscan import BatchedDBSCAN

    max_number_of_tracks = 232
    max_number_of_tracks_power_2 = 256
    max_number_of_tracks_log_2 = 8

    for bs in [10, 20, 40, 50, 80, 52, 102, 214, 500]:
        if bs > 10:
            break
        batch_size = bs
        eps = 0.15

        storage_path = "/media/lucas/QS/binaries-trk-100/"
        storage_acc = "/media/lucas/QS/accelerated_results/"
        sum_diff = 0
        diff = -999 * np.ones([100], dtype=np.float32)
        for i in range(100):
            print(f"i: {i}, bs: {bs}") 
            z0_file = storage_path + f"b-{i}-trk-z0.bin"
            pt_file = storage_path + f"b-{i}-trk-pt.bin"
            mc_file = storage_path + f"b-{i}-trk-mc.bin"

            z0 = np.fromfile(z0_file, dtype=np.float32)
            pt = np.fromfile(pt_file, dtype=np.float32)
            mc = np.fromfile(mc_file, dtype=np.float32)

            db = BatchedDBSCAN(
                z0, pt, eps, batch_size, max_number_of_tracks, False, False
            )

            # db.fitsklearn()
            db.fit()

            # Accelerated results
            z0_t = np.loadtxt(storage_acc + f"pv-z0-{i}.txt", dtype=np.float32)
            pt_t = np.loadtxt(storage_acc + f"pv-pt-{i}.txt", dtype=np.float32)

            print(db.z0_pv_skl, z0_t)

            diff[i] = db.z0_pv_skl - z0_t

        print("max: ", max(diff))
        print("min: ", min(diff))
        print("mean: ", np.mean(diff))
        print("median: ", np.median(diff))
        print("std: ", np.std(diff))
