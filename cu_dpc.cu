#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <iostream>
#include <algorithm>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <assert.h>
#include <thrust/sort.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define MAX_CHAR_PER_LINE 128

__device__ float euclid_dist(int numCoords, float *objects1, int i, int j) {
    float ans = 0.0;
    for (int k = 0; k < numCoords; k++) {
        float x = objects1[i * numCoords + k];
        float y = objects1[j * numCoords + k];
        ans += (x - y) * (x - y);
    }
    return sqrt(ans);
}

__host__ void argsort_gamma(int numObjs, int *gamma_index, float *d_gamma) {
    int *h_gamma = (int*)malloc(sizeof(int) * numObjs);
    cudaMemcpy(h_gamma, d_gamma, sizeof(int) * numObjs, cudaMemcpyDeviceToHost);
    for (int i = 0; i < numObjs; i++) {
        gamma_index[i] = i;
    }
    std::sort(gamma_index, gamma_index + numObjs, [&h_gamma](int pos1, int pos2) { return (h_gamma[pos1] > h_gamma[pos2]); });
    free(h_gamma);
}

__host__ void argsort_rho(int numObjs, int *d_rho, int *d_rho_index_sorted) {
    int *index = (int*)malloc(sizeof(int) * numObjs);
    int *h_rho = (int*)malloc(sizeof(int) * numObjs);
    cudaMemcpy(h_rho, d_rho, sizeof(int) * numObjs, cudaMemcpyDeviceToHost);
    for (int i = 0; i < numObjs; i++) {
        index[i] = i;
    }
    std::sort(index, index + numObjs, [&h_rho](int pos1, int pos2) { return (h_rho[pos1] > h_rho[pos2]); });
    cudaMemcpy(d_rho_index_sorted, index, sizeof(int) * numObjs, cudaMemcpyHostToDevice);
    free(index);
    free(h_rho);
}

__global__ void get_distance(int numCoords, int numObjs, float *objects1, float *distance1) {
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    for (int i = tid_y; i < numObjs; i += blockDim.y) {
        for (int j = tid_x; j < numObjs; j += blockDim.x) {
            if (j >= i) {
                float num = euclid_dist(numCoords, objects1, i, j);
                distance1[i * numObjs + j] = num;
                distance1[j * numObjs + i] = num;
            }
        }
    }
}

__global__ void get_rho(int numObjs, float d_c, int *rho, float *distance1) {
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    for (int i = tid_y * blockDim.x + tid_x; i < numObjs; i += blockDim.x * blockDim.y) {
        rho[i] = -1;
        for (int j = 0; j < numObjs; j++) {
            if (distance1[i * numObjs + j] < d_c) {
                rho[i]++;
            } 
        }
    }
}

__global__ void get_delta(int numObjs, int *nearest, int *rho_index_sorted, float *delta, float *distance1) {
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    for (int seq = tid_y * blockDim.x + tid_x; seq < numObjs; seq += blockDim.x * blockDim.y) {
        int i = rho_index_sorted[seq];
        if (seq == 0)
            continue;
        float minn = MAXFLOAT;
        int nearest_index = 0;
        for (int k = 0; k < seq; k++) {
            float dist = distance1[i * numObjs + rho_index_sorted[k]];
            if (dist < minn) {
                minn = dist;
                nearest_index = k;
            }
        }
        delta[i] = minn;
        nearest[i] = rho_index_sorted[nearest_index];
    }
    if (tid_x == 0) {
        float maxx = 0;
        int idx = rho_index_sorted[0];
        for (int j = 0; j < numObjs; j++) {
            maxx = max(maxx, distance1[idx * numObjs + j]);
        }
        delta[idx] = maxx;
    }
}

__global__ void get_gamma(int numObjs, int *rho, float *delta, float *gamma) {
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    for (int i = tid_y * blockDim.x + tid_x; i < numObjs; i += blockDim.x * blockDim.y) {
        gamma[i] = rho[i] * delta[i];
    }
}

__host__ void fit(int numClusters, int numCoords, int numObjs, float d_c,
                  int *d_rho, int *d_nearest, int *gamma_index, int *category, int *center,
                  int *center_index, int *d_rho_index_sorted, float *d_delta, float *d_gamma,
                  float *d_objects1, float *objects1, float *d_distance1, float *clusters1) {
    dim3 blockSize(32, 32);
    get_distance<<<1, blockSize>>>(numCoords, numObjs, d_objects1, d_distance1);
    cudaDeviceSynchronize();
    get_rho<<<1, blockSize>>>(numObjs, d_c, d_rho, d_distance1);
    cudaDeviceSynchronize();
    argsort_rho(numObjs, d_rho, d_rho_index_sorted);
    get_delta<<<1, blockSize>>>(numObjs, d_nearest, d_rho_index_sorted, d_delta, d_distance1);
    cudaDeviceSynchronize();
    get_gamma<<<1, blockSize>>>(numObjs, d_rho, d_delta, d_gamma);
    cudaDeviceSynchronize();
    argsort_gamma(numObjs, gamma_index, d_gamma);

    int *nearest = (int*)malloc(sizeof(int) * numObjs);
    cudaMemcpy(nearest, d_nearest, sizeof(int) * numObjs, cudaMemcpyDeviceToHost);
    int *rho_index_sorted = (int*)malloc(sizeof(int) * numObjs);
    cudaMemcpy(rho_index_sorted, d_rho_index_sorted, sizeof(int) * numObjs, cudaMemcpyDeviceToHost);

    int current_category = 1;
    for (int seq = 0; seq < numObjs; seq++) {
        int i = gamma_index[seq];
        if (seq >= numClusters)
            break;
        category[i] = current_category;
        center[i] = current_category;
        center_index[seq] = i;
        for (int j = 0; j < numCoords; j++) {
            clusters1[seq * numCoords + j] = objects1[i * numCoords + j];
        }
        current_category++;
    }
    for (int seq = 0; seq < numObjs; seq++) {
        int i = rho_index_sorted[seq];
        if (category[i] == 0)
            category[i] = category[nearest[i]];
    }

    free(nearest);
    free(rho_index_sorted);
}

void printDeviceProp(const cudaDeviceProp prop) {
    printf("Device Name: %s\n", prop.name);
    printf("totalGlobalMem: %.0f MBytes---%ld Bytes\n", (float)prop.totalGlobalMem/1024/1024, prop.totalGlobalMem);
    printf("sharedMemPerBlock: %lu\n", prop.sharedMemPerBlock);
    printf("regsPerBolck: %d\n", prop.regsPerBlock);
    printf("warpSize: %d\n", prop.warpSize);
    printf("memPitch: %lu\n", prop.memPitch);
    printf("maxTreadsPerBlock: %d\n", prop.maxThreadsPerBlock);
    printf("maxThreadsDim[0-2]: %d %d %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("maxGridSize[0-2]: %d %d %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("totalConstMem: %lu\n", prop.totalConstMem);
    printf("major.minor: %d.%d\n", prop.major, prop.minor);
    printf("clockRate: %d\n", prop.clockRate);
    printf("textureAlignment: %lu\n", prop.textureAlignment);
    printf("deviceOverlap: %d\n", prop.deviceOverlap);
    printf("multiProcessorCount: %d\n", prop.multiProcessorCount);
    printf("===========================\n\n");
}

float **file_read(char *filename, int *numObjs, int *numCoords) {
    float **objects;
    int i, j, len;

    /* input file is in ASCII format */
    FILE *infile;
    char *line, *ret;
    int lineLen;

    if ((infile = fopen(filename, "r")) == NULL) {
        fprintf(stderr, "Error: no such file (%s)\n", filename);
        return NULL;
    }

    /* first find the number of objects */
    lineLen = MAX_CHAR_PER_LINE;
    line = (char *)malloc(lineLen);
    assert(line != NULL);

    (*numObjs) = 0;
    while (fgets(line, lineLen, infile) != NULL) {
        /* check each line to find the max line length */
        while (strlen(line) == lineLen - 1) {
            /* this line read is not complete */
            len = strlen(line);
            fseek(infile, -len, SEEK_CUR);

            /* increase lineLen */
            lineLen += MAX_CHAR_PER_LINE;
            line = (char *)realloc(line, lineLen);
            assert(line != NULL);

            ret = fgets(line, lineLen, infile);
            assert(ret != NULL);
        }

        if (strtok(line, " \t\n") != 0)
            (*numObjs)++;
    }
    rewind(infile);

    /* find the no. coordinates of each object */
    (*numCoords) = 0;
    while (fgets(line, lineLen, infile) != NULL) {
        if (strtok(line, " \t\n") != 0) {
            /* ignore the id (first coordiinate): numCoords = 1; */
            while (strtok(NULL, " ,\t\n") != NULL)
                (*numCoords)++;
            break; /* this makes read from 1st object */
        }
    }
    rewind(infile);

    /* allocate space for objects[][] and read all objects */
    len = (*numObjs) * (*numCoords);
    objects = (float **)malloc((*numObjs) * sizeof(float *));
    assert(objects != NULL);
    objects[0] = (float *)malloc(len * sizeof(float));
    assert(objects[0] != NULL);
    for (i = 1; i < (*numObjs); i++)
        objects[i] = objects[i - 1] + (*numCoords);

    i = 0;
    /* read all objects */
    while (fgets(line, lineLen, infile) != NULL) {
        if (strtok(line, " \t\n") == NULL)
            continue;
        for (j = 0; j < (*numCoords); j++) {
            objects[i][j] = atof(strtok(NULL, " ,\t\n"));
        }
        i++;
    }
    assert(i == *numObjs);

    fclose(infile);
    free(line);

    return objects;
}

int file_write(char *filename, int numClusters, int numObjs, int numCoords, float **centers, int *category) {
    FILE *fptr;
    int i, j;
    char outFileName[1024];

    /* output: the coordinates of the cluster centres ----------------------*/
    sprintf(outFileName, "%s.centers", filename);
    fptr = fopen(outFileName, "w");
    for (i = 0; i < numClusters; i++)
    {
        fprintf(fptr, "%d ", i);
        for (j = 0; j < numCoords; j++)
            fprintf(fptr, "%f ", centers[i][j]);
        fprintf(fptr, "\n");
    }
    fclose(fptr);

    /* output: the closest cluster centre to each of the data points --------*/
    sprintf(outFileName, "%s.category", filename);
    fptr = fopen(outFileName, "w");
    for (i = 0; i < numObjs; i++)
        fprintf(fptr, "%d %d\n", i, category[i]);
    fclose(fptr);

    return 1;
}

static void usage(char *argv0, float d_c) {
    const char *help =
        "Usage: %s [switches] -i filename -n num_clusters\n"
        "       -i filename    : file containing data to be clustered\n"
        "       -n num_clusters: number of clusters (K must > 1)\n"
        "       -t d_c         : d_c value (default %.4f)\n"
        "       -h             : print this help information\n";
    fprintf(stderr, help, argv0, d_c);
    exit(-1);
}

int main(int argc, char **argv) {
    int opt;
    int numClusters, numCoords, numObjs;
    float d_c = 0.05;
    char *filename = NULL;
    int *d_rho, *d_nearest, *gamma_index, *category, *center;
    int *center_index, *d_rho_index_sorted;
    float *d_delta, *d_gamma;
    float *objects1, *d_objects1, *d_distance1, *clusters1;


    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printDeviceProp(prop);
    cudaSetDevice(0);

    while ((opt = getopt(argc, argv, "i:n:t:h")) != EOF) {
        switch (opt) {
        case 'i':
            filename = optarg;
            break;
        case 't':
            d_c = atof(optarg);
            break;
        case 'n':
            numClusters = atoi(optarg);
            break;
        case 'h':
        default:
            usage(argv[0], d_c);
            break;
        }
    }

    float **objects = file_read(filename, &numObjs, &numCoords);

    objects1 = (float*)malloc(sizeof(float) * numObjs * numCoords);
    for (int i = 0; i < numObjs; i++) {
        for (int j = 0; j < numCoords; j++) {
            objects1[i * numCoords + j] = objects[i][j];
        }
    }

    //===================步骤1===================
    // Allocate memory on GPU

    gamma_index = (int*)malloc(sizeof(int) * numObjs);
    category = (int*)malloc(sizeof(int) * numObjs);
    center = (int*)malloc(sizeof(int) * numObjs);
    center_index = (int*)malloc(sizeof(int) * numClusters);
    clusters1 = (float*)malloc(sizeof(float) * numClusters * numCoords);
    
    cudaMalloc((void**)&(d_rho), sizeof(int) * numObjs);
    cudaMalloc((void**)&(d_nearest), sizeof(int) * numObjs);
    cudaMalloc((void**)&(d_rho_index_sorted), sizeof(int) * numObjs);

    cudaMalloc((void**)&(d_delta), sizeof(float) * numObjs);
    cudaMalloc((void**)&(d_gamma), sizeof(float) * numObjs);

    cudaMalloc((void**)&(d_objects1), sizeof(float) * numObjs * numCoords);
    cudaMalloc((void**)&(d_distance1), sizeof(float) * numObjs * numObjs);


    //===================步骤2===================
    // copy operator to GPU
    cudaMemcpy(d_objects1, objects1, sizeof(float) * numObjs * numCoords, cudaMemcpyHostToDevice);
    
    //===================步骤3===================
    // GPU do the work, CPU waits
    float time = 0.0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    fit(numClusters, numCoords, numObjs, d_c,
        d_rho, d_nearest, gamma_index, category, center,
        center_index, d_rho_index_sorted, d_delta, d_gamma,
        d_objects1, objects1, d_distance1, clusters1);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    
    //===================步骤4===================
    // Get results from the GPU

    printf("\n===== DensityPeaksClustering (Cuda version) =====\n");

    printf("Input file:       %s\n", filename);
    printf("numObjs         = %d\n", numObjs);
    printf("numCoords       = %d\n", numCoords);
    printf("numClusters     = %d\n", numClusters);
    printf("threshold d_c   = %.4f\n", d_c);
    printf("clustering time = %.4f milliseconds\n", time);
    printf("\n===== clustering centers =====\n");

    for (int i = 0; i < numClusters; i++) {
        printf("clustering centers %d: ", i);
        for (int j = 0; j < numCoords; j++) {
            printf("%f ", clusters1[i * numCoords + j]);
        }
        printf("\n");
    }

    float **clusters = (float **)malloc(sizeof(float*) * numClusters);
    for (int i = 0; i < numClusters; i++) {
        clusters[i] = (float*)malloc(sizeof(float) * numCoords);
    }
    for (int i = 0; i < numClusters; i++) {
        for (int j = 0; j < numCoords; j++) {
            clusters[i][j] = clusters1[i * numCoords + j];
        }
    }

    /* output: the coordinates of the cluster centres */
    file_write(filename, numClusters, numObjs, numCoords, clusters, category);

    //===================步骤5===================
    // Free the memory
    free(objects1);
    free(gamma_index);
    free(category);
    free(center);
    free(center_index);
    free(clusters1);

    cudaFree(d_rho);
    cudaFree(d_nearest);
    cudaFree(d_rho_index_sorted);

    cudaFree(d_delta);
    cudaFree(d_gamma);

    cudaFree(d_objects1);
    cudaFree(d_distance1);

    cudaDeviceReset();
    return 0;
}