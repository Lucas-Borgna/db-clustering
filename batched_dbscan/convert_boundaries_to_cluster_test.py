import pandas as pd
import numpy as np


boundaries = np.load('boundaries.npy')

def initialize_clusters(max_n_clusters: int) -> np.array:
    clusters = np.zeros((max_n_clusters, 4))
    
    clusters[:, 1] = 21
    clusters[:, 2] = 21
    
    return clusters
    
def convert_boundaries_to_clusters(boundaries:np.array):
    bound_i = 0
    cluster_j = 0
    n_boundaries = boundaries.shape[0]
    z0_low_idx = 1
    z0_high_idx = 2
    pt_idx = 0
    noise_idx = 3 
    clusters = initialize_clusters(n_boundaries)
    
    for _ in range(n_boundaries):
        print("bound_i: ", bound_i, "cluster_j: ", cluster_j)
        # if boundaries[bound_i, 4] == 21:
            # break
        noise = boundaries[bound_i, -1]
        
        if noise:
            z0_low = boundaries[bound_i, 4]
            z0_high = boundaries[bound_i, 5]
            pt_sum = boundaries[bound_i, 2] - boundaries[bound_i, 1]
            
            clusters[cluster_j, z0_low_idx] = z0_low
            clusters[cluster_j, z0_high_idx] = z0_high
            clusters[cluster_j, pt_idx] = pt_sum
            clusters[cluster_j, noise_idx] = noise
            bound_i += 1
            cluster_j += 1
        else:
            z0_low = boundaries[bound_i, 4]
            z0_high = boundaries[bound_i + 1, 4]
            pt_sum = boundaries[bound_i + 1, 2] - boundaries[bound_i, 1]

            clusters[cluster_j, z0_low_idx] = z0_low
            clusters[cluster_j, z0_high_idx] = z0_high
            clusters[cluster_j, pt_idx] = pt_sum
            clusters[cluster_j, noise_idx] = noise
            
            bound_i+=2
            cluster_j+=1
        
        return clusters

print(boundaries)

clusters = convert_boundaries_to_clusters(boundaries)

print(clusters) 