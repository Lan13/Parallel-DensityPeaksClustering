# Parallel-DensityPeaksClustering
USTC 2023 Spring Parallel Computing B Course Work: DensityPeaksClustering written in OpenMP &amp; Cuda



This work is inspired by serban's kmeans [serban/kmeans: A CUDA implementation of the k-means clustering algorithm (github.com)](https://github.com/serban/kmeans). Because my machine learning course project has done density peaks clustering algorithm, I would like to try it with OpenMP and Cuda.



`cat.txt`: origin dataset with 31 clusters (3100 data elements copied to 40300)

`catn.txt`: dataset preprocess by minmaxscaler (min-max normalization)

**The dimension of this dataset is 2 for easy visualization, if your dataset is not, then you need to modify part of codes.**



Here's usage:

```
"Usage: %s [switches] -i filename -n num_clusters\n"
        "       -i filename    : file containing data to be clustered\n"
        "       -n num_clusters: number of clusters (K must > 1)\n"
        "       -t d_c         : d_c value (default %.4f)\n"
        "       -h             : print this help information\n";
```



Here's some sample benchmark output for the dataset on an Intel(R) Xeon(R) Gold 6154 CPU @ 3.00GHz machine with an NVIDIA GeForce RTX 2080 Ti card. (Run the benchmark.sh)

```
========================== DensityPeaksClustering =============================
-------------------------------------------------------------------------------
seqTime = 58581.9130ms  ompTime = 37975.4223ms  speedup = 1.54x  (OpenMP Version)
seqTime = 58581.9130ms  omp2Time = 36567.8964ms  speedup = 1.60x  (Argsort Version)
seqTime = 58581.9130ms  cudaTime = 2838.2395ms  speedup = 20.64x  (Cuda Version)
seqTime = 58581.9130ms  cuda2Time = 548.4465ms  speedup = 106.81x  (Optimized Version)
seqTime = 58581.9130ms  cuda3Time = 631.6838ms  speedup = 92.73x  (Texture Version)
seqTime = 58581.9130ms  cuda4Time = 423.0862ms  speedup = 138.46x  (Shared Version)
-------------------------------------------------------------------------------
```



You can view the results of clustering by the following method:

```python
cat = pd.read_csv("./cat.txt", header=None, sep=" ")
cat_1 = np.array((cat - cat.min()) / (cat.max() - cat.min()))
centers = pd.read_csv("./catn.txt.centers", header=None, sep=" ")
labels = pd.read_csv("./catn.txt.category", header=None, sep=" ")

plt.scatter(cat_1[:, 0], cat_1[:, 1], c=labels[1])
plt.scatter(centers[1], centers[2], marker="x" , c="red")
```

results of clustering:

<img src="./clusters.png" alt="clusters" style="zoom:67%;" />
