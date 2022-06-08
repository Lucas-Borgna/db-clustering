import numpy as np

from acceleratedDBSCAN import AccDBSCAN
from batchedDBSCAN import BatchedDBSCAN
from batchedDBSCAN_v2 import BatchedDBSCANV2


import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="run accelerated dbscan")
    parser.add_argument(
        "-b",
        "--batch-size",
        help="batch size to use when running",
        default=50,
        type=int,
    )
    args = vars(parser.parse_args())

    max_number_of_tracks = 232
    max_number_of_tracks_power_2 = 256
    max_number_of_tracks_log_2 = 8
    batch_size = args["batch_size"]
    eps = 0.15

    z0_file = "/media/lucas/QS/binaries-trk/OldKF_TTbar_170K_quality-1-trk-z0.bin"
    pt_file = "/media/lucas/QS/binaries-trk/OldKF_TTbar_170K_quality-1-trk-pt.bin"
    z0 = np.fromfile(z0_file, dtype=np.float32)
    pt = np.fromfile(pt_file, dtype=np.float32)

    # z0 = z0[0:40]
    # pt = pt[0:40]
    print("--------------------------")
    db = BatchedDBSCAN(z0, pt, eps, batch_size, max_number_of_tracks, True)

    db.fit()
    # print("--------------------------")
    # dbb2 = BatchedDBSCANV2(z0, pt, eps, batch_size, max_number_of_tracks, True)

    # dbb2.fit()
    print("--------------------------")
    db2 = AccDBSCAN(z0, pt, eps, max_number_of_tracks, True)
    db2.fit()
    print(db2.pv_z0, db2.pv_pt)

    # print(db.pv_z0)
