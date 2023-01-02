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
    eps = 0.15 
    verbose = False
    save_intermediate = False
    
    storage_path = "/media/lucas/QS/binaries-trk-100/"
    storage_acc = "/media/lucas/QS/accelerated_results/"
    total_count = 0
    fail_count = 0
    for i in range(49):
        z0_file = storage_path + f"b-{i}-trk-z0.bin"
        pt_file = storage_path + f"b-{i}-trk-pt.bin"
        mc_file = storage_path + f"b-{i}-trk-mc.bin"

        z0 = np.fromfile(z0_file, dtype=np.float32)
        pt = np.fromfile(pt_file, dtype=np.float32)
        mc = np.fromfile(mc_file, dtype=np.float32)
        z0_t = np.loadtxt(storage_acc + f"pv-z0-{i}.txt", dtype=np.float32)
        pt_t = np.loadtxt(storage_acc + f"pv-pt-{i}.txt", dtype=np.float32)
        min_bs = 999 * np.ones(100)
        
        print(z0[0], pt[0], z0_t, pt_t)
        
        for bs in range(1, max_number_of_tracks+5):
            # print(f"i: {i}, bs: {bs}")
            total_count +=1

            try:
                db = BatchedDBSCAN(z0, pt, eps, bs, max_number_of_tracks, verbose, save_intermediate)
                db.fit()

#                 z0_pdiff = 100*(np.abs(db.z0_pv - z0_t)/z0_t)
#                 pt_pdiff = 100*(np.abs(db.max_pt - pt_t)/pt_t)
                
#                 if (z0_pdiff <= 0.05) and (pt_pdiff <=0.05):
#                     min_bs[i] = bs
#                     break
            except:
                fail_count+=1
                print(f"file {i}, batch_size {bs} failed")
                
    
    print(total_count, fail_count)
#     print(f"mean: {np.mean(min_bs)}")
#     print(f"min: {np.min(min_bs)}")
#     print(f"max: {np.max(min_bs)}")
    
