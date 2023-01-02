import numpy as np
import math
import pandas as pd
import itertools
import copy
import matplotlib.pyplot as plt

np.random.seed(42)


if __name__ == "__main__":

    max_number_of_tracks = 232
    max_number_of_tracks_power_2 = 256
    max_number_of_tracks_log_2 = 8
    eps = 0.15
    verbose = False
    save_intermediate = False

    z0_file = "/media/lucas/QS/binaries-trk/OldKF_TTbar_170K_quality-1-trk-z0.bin"
    pt_file = "/media/lucas/QS/binaries-trk/OldKF_TTbar_170K_quality-1-trk-pt.bin"
    z0 = np.fromfile(z0_file, dtype=np.float32)
    pt = np.fromfile(pt_file, dtype=np.float32)

    from batched_dbscan import BatchedDBSCAN
    batch_sizes = []
    z0s = []
    pts = []
    z0_truth = -2.2265625
    pt_truth = 142.0233987569809
    
    for i in range(1, max_number_of_tracks+5):
        
        batch_size = i
        
        try:
            
            db = BatchedDBSCAN(z0, pt, eps, batch_size, max_number_of_tracks, verbose, save_intermediate)

            db.fit()
            print(f"z0 = {db.z0_pv}, pt = {db.max_pt}")
            z0s.append(db.z0_pv)
            pts.append(db.max_pt)            
            batch_sizes.append(i)

            
        except:
            print(f"batch_size {i} failed")
    
    z0_pdiff = 100 * ((np.array(z0s) - z0_truth)/z0_truth)
    pt_pdiff = 100 * ((np.array(pts) - pt_truth)/pt_truth)
    
    for i in range(z0_pdiff.shape[0]):
        if (np.abs(z0_pdiff[i]) < 0.1) and (np.abs(pt_pdiff[i]) < 0.1):
            min_batch_size = i
            break
            
    print(min_batch_size)
            

    
    plt.plot(batch_sizes, z0s, color = 'green',lw=2)
    plt.axhline(z0_truth, ls='--',color='grey')
    plt.title(f'min batch_size = {min_batch_size}')
    plt.xlabel('batch size')
    plt.ylabel('z0 pv [cm]')
    plt.savefig('z0_vs_batch_size.png')
    plt.clf()
    
    plt.plot(batch_sizes, z0_pdiff, color = 'green',lw=2)
    plt.axhline(0,ls='--',color='grey')
    plt.title(f'min batch_size = {min_batch_size}')
    plt.xlabel('batch size')    
    plt.ylabel("z0 percentage difference [%]")
    plt.savefig('z0_vs_batch_size_pdiff.png')
    plt.clf()
    
    plt.plot(batch_sizes, pts, color = 'crimson',lw=2)
    plt.axhline(pt_truth, ls='--',color='grey')
    plt.title(f'min batch_size = {min_batch_size}')
    plt.xlabel('batch size')
    plt.ylabel('pt sum pv [GeV]')
    plt.savefig('pt_vs_batch_size.png')
    plt.clf()
    
    plt.plot(batch_sizes, pt_pdiff, color = 'crimson',lw=2)
    plt.axhline(0,ls='--',color='grey')
    plt.title(f'min batch_size = {min_batch_size}')
    plt.xlabel('batch size')    
    plt.ylabel("pt percentage difference [%]")
    plt.savefig('pt_vs_batch_size_pdiff.png')
    plt.clf()
    
    # import json
    # with open('merged_list.json', 'w') as f:
    #     json.dump(db.merged_list, f, indent=4)
    
