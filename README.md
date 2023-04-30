# Parallel-DensityPeaksClustering
USTC 2023 Spring Parallel Computing B Course Work: DensityPeaksClustering written in OpenMP &amp; Cuda



This work is inspired by serban's kmeans [serban/kmeans: A CUDA implementation of the k-means clustering algorithm (github.com)](https://github.com/serban/kmeans). Because my machine learning course project has done density peaks clustering algorithm, I would like to try it with OpenMP and Cuda.



`cat.txt`: origin dataset with 31 clusters (3100 data elements copied to 40300)

`catn.txt`: dataset proprocess by minmaxscaler (min-max normalization)



Here's some sample benchmark output for the dataset on an Intel(R) Xeon(R) Gold 6154 CPU @ 3.00GHz machine with an NVIDIA GeForce RTX 2080 Ti card. 

```
===================== DensityPeaksClustering ========================
---------------------------------------------------------------------
seqTime = 59701.8310ms  ompTime = 34327.7565ms  speedup = 1.73x
seqTime = 59701.8310ms  cudaTime = 2780.1560ms  speedup = 21.47x
seqTime = 59701.8310ms  cuda2Time = 558.1675ms  speedup = 106.96x
---------------------------------------------------------------------
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
