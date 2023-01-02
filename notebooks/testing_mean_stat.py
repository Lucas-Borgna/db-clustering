import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


trk = pd.read_pickle("/home/raichu/data/trk_processed_25k.pkl")
mc = pd.read_pickle("/home/raichu/data/mc_25k.pkl")

from primaryvertexingtools import PrimaryVertexing, PerformanceMetrics

fh_bins = np.linspace(-15,15,256)

mydict = {"track_data":trk, "truth_data":mc,"fh_bins":fh_bins, "test_run":False}

PV = PrimaryVertexing(mydict)
    
PV.run_fh()
eff=[]
for ep in [0.04, 0.06, 0.10, 0.15]:
        
    PV.run_dbscan(eps=ep, stat='mean')
    
    pm = PerformanceMetrics(PV)
    
    eff.append(pm.pv_efficiency((pm.z0_gen - pm.z0_reco_db_cor), display=True))
    
print(eff)