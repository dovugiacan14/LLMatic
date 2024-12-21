import math
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans

default_params = {
    "cvt_samples": 25000, # control quality CVT
    "batch_size": 100,    # evaluate in batches to parallel 
    "random_init": 0.1,   # percentage of niches to be filled before starting
    "random_init_batch": 100, # batch for random initialization 
    "dump_period": 10000,     # period to write results 
    "parallel": True,      # use several cores 
    "cvt_use_cache": True, # we cache the result of CVT and reuse
    # min-max of parameters 
    "min": 0, 
    "max": 1, 
    # only useful if you use the 'iso_dd' variation operator 
    "iso_sigma": 0.01, 
    "line_sigma": 0.02
}

def __centroids_filename(k, dim):
    return "centroids_" + str(k) + "_" + str(dim) + ".dat"

def __write_centroids(centroids):
    k = centroids.shape[0]
    dim = centroids.shape[1]
    filename = __centroids_filename(k, dim)
    with open(filename, 'w') as f: 
        for p in centroids:
            for item in p: 
                f.write(str(item) + " ")
            f.write("\n")

def cvt(k, dim, samples, cvt_use_cache= True):
    fname = __centroids_filename(k, dim)
    if cvt_use_cache:
        if Path(fname).is_file():
            print("WARNING: using cached CVT:", fname)
            return np.loadtxt(fname)
    print("Computing CVT (this can take a while...):", fname)
    
    x = np.random.rand(samples, dim)
    k_means = KMeans(init= 'k-means++', n_clusters= k, n_init= 1, verbose= 1)
    k_means.fit(x)
    __write_centroids(k_means.cluster_centers_)
    return k_means.cluster_centers_

def make_hashable(array):
    return tuple(map(float, array))

def __save_archieve(archive, gen, name= "net"):
    def write_array(a, f):
        for i in a: 
            f.write(str(i) + " ")
    filename = name + "archive_" + str(gen) + '.dat'
    with open(filename, 'w') as f: 
        for k in archive.values():
            f.write(k.net_path + " ")
            f.write(str(k.fitness) + ' ')
            write_array(k.centroid, f)
            f.write(str(k.curiosity) + " ")
            f.write(str(k.desc[0]) + " ")
            f.write(str(k.desc[1]) + " ")
            f.write("\n")

class Species:
    def __init__(self, x, desc, fitness, net_path, centroid=None):
        self.x = x
        self.desc = desc
        self.fitness = fitness
        self.centroid = centroid
        self.curiosity = 0
        self.net_path = net_path