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
#include <omp.h>

#define MAX_CHAR_PER_LINE 128
#define NUM_THREADS 72

class DensityPeaksClustering {
    public:
        int numObjs;
        int numCoords;
        int numClusters;

        float d_c;

        int *rho;
        int *nearest;
        int *gamma_index;
        int *category;
        int *center;
        int *center_index;
        int *rho_index_sorted;


        float *delta;
        float *gamma;

        float **objects;
        float **distance;
        float **clusters;

        DensityPeaksClustering(int numObjs, int numCoords, int numClusters, float **objects, float d_c);
        ~DensityPeaksClustering();
        float euclid_dist(int numCoords, float *coord1, float *coord2);
        int *argsort(int numObjs, float *arrs);
        int *argsort(int numObjs, int *arrs);
        void get_distance();
        void get_rho();
        void get_delta();
        void get_gamma();
        void fit();
};

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

DensityPeaksClustering::DensityPeaksClustering(int numObjs, int numCoords, int numClusters, float **objects, float d_c) {
    this->numObjs = numObjs;
    this->numCoords = numCoords;
    this->numClusters = numClusters;
    this->d_c = d_c;
    this->objects = objects;

    this->rho = (int*)malloc(sizeof(int) * numObjs);
    this->nearest = (int*)malloc(sizeof(int) * numObjs);
    this->gamma_index = (int*)malloc(sizeof(int) * numObjs);
    this->category = (int*)malloc(sizeof(int) * numObjs);
    this->center = (int*)malloc(sizeof(int) * numObjs);
    this->center_index = (int*)malloc(sizeof(int) * numClusters);
    this->rho_index_sorted = (int*)malloc(sizeof(int) * numObjs);

    this->delta = (float*)malloc(sizeof(float) * numObjs);
    this->gamma = (float*)malloc(sizeof(float) * numObjs);
    
    this->distance = (float **)malloc(sizeof(float*) * numObjs);
    for (int i = 0; i < numObjs; i++) {
        this->distance[i] = (float*)malloc(sizeof(float) * numObjs);
    }

    this->clusters = (float **)malloc(sizeof(float*) * numClusters);
    for (int i = 0; i < numClusters; i++) {
        this->clusters[i] = (float*)malloc(sizeof(float) * numCoords);
    }
}

DensityPeaksClustering::~DensityPeaksClustering() {
    free(rho);
    free(nearest);
    free(gamma_index);
    free(category);
    free(center);
    free(center_index);
    free(rho_index_sorted);
    
    free(delta);
    free(gamma);

    for (int i = 0; i < numObjs; i++) {
        free(distance[i]);
    }
    free(distance);

    for (int i = 0; i < numClusters; i++) {
        free(clusters[i]);
    }
    free(clusters);
}

float DensityPeaksClustering::euclid_dist(int numCoords, float *coord1, float *coord2) {
    float ans = 0.0;
    assert(numCoords <= 72);
    omp_set_num_threads(numCoords);
    #pragma omp parallel for reduction(+:ans) num_threads(numCoords) shared(coord1, coord2, numCoords)
    for (int i = 0; i < numCoords; i++) {
        ans += (coord1[i] - coord2[i]) * (coord1[i] - coord2[i]);
    }
    return sqrt(ans);
}

int *DensityPeaksClustering::argsort(int numObjs, float *arrs) {
    int *index = (int*)malloc(sizeof(int) * numObjs);
    omp_set_num_threads(NUM_THREADS);
    #pragma omp parallel num_threads(NUM_THREADS) shared(index, numObjs)
    {
        int tid = omp_get_thread_num();
        #pragma omp parallel for
        for (int i = tid; i < numObjs; i += NUM_THREADS) {
            index[i] = i;
        }
    }
    std::sort(index, index + numObjs, [&arrs](int pos1, int pos2) { return (arrs[pos1] > arrs[pos2]); });
    return index;
}

int *DensityPeaksClustering::argsort(int numObjs, int *arrs) {
    int *index = (int*)malloc(sizeof(int) * numObjs);
    omp_set_num_threads(NUM_THREADS);
    #pragma omp parallel num_threads(NUM_THREADS) shared(index, numObjs)
    {
        int tid = omp_get_thread_num();
        #pragma omp parallel for
        for (int i = tid; i < numObjs; i += NUM_THREADS) {
            index[i] = i;
        }
    }
    std::sort(index, index + numObjs, [&arrs](int pos1, int pos2) { return (arrs[pos1] > arrs[pos2]); });
    return index;
}

void DensityPeaksClustering::get_distance() {
    omp_set_num_threads(NUM_THREADS);
    #pragma omp parallel num_threads(NUM_THREADS) shared(distance, objects, numObjs)
    {
        int tid = omp_get_thread_num();
        #pragma omp parallel for
        for (int i = tid; i < numObjs; i += NUM_THREADS) 
        {
            for (int j = i; j < numObjs; j++) 
            {
                distance[i][j] = euclid_dist(numCoords, objects[i], objects[j]);
                distance[j][i] = distance[i][j];
            }
        }
    }
}

void DensityPeaksClustering::get_rho() {
    omp_set_num_threads(NUM_THREADS);
    #pragma omp parallel num_threads(NUM_THREADS) shared(distance, rho, numObjs, d_c)
    {
        int tid = omp_get_thread_num();
        #pragma omp parallel for
        for (int i = tid; i < numObjs; i += NUM_THREADS) {
            rho[i] = 0;
            for (int j = 0; j < numObjs; j++) {
                if (distance[i][j] < d_c) {
                    rho[i]++;
                }
            }
            rho[i] -= 1;    // exclude itself
        }
    }
}

void DensityPeaksClustering::get_delta() {
    rho_index_sorted = argsort(numObjs, rho);
    omp_set_num_threads(NUM_THREADS);
    #pragma omp parallel num_threads(NUM_THREADS) shared(rho_index_sorted, distance, delta, nearest, numObjs)
    {
        int tid = omp_get_thread_num();
        #pragma omp parallel for
        for (int seq = tid; seq < numObjs; seq += NUM_THREADS) {
            int i = rho_index_sorted[seq];
            if (seq == 0)
                continue;
            int *j = (int*)malloc(sizeof(int) * seq);
            memcpy(j, rho_index_sorted, sizeof(int) * seq);
            float minn = MAXFLOAT;
            int nearest_index = 0;
            for (int k = 0; k < seq; k++) {
                if (distance[i][j[k]] < minn) {
                    minn = distance[i][j[k]];
                    nearest_index = k;
                }
            }

            delta[i] = minn;
            nearest[i] = j[nearest_index];
        }
    }
    
    float maxx = 0;
    omp_set_num_threads(NUM_THREADS);
    #pragma omp parallel num_threads(NUM_THREADS) shared(distance, rho_index_sorted, numObjs)
    {
        int tid = omp_get_thread_num();
        #pragma omp parallel for reduction(max:maxx)
        for (int j = tid; j < numObjs; j += NUM_THREADS) {
            maxx = std::max(maxx, distance[rho_index_sorted[0]][j]);
        }
    }

    delta[rho_index_sorted[0]] = maxx;
}

void DensityPeaksClustering::get_gamma() {
    omp_set_num_threads(NUM_THREADS);
    #pragma omp parallel num_threads(NUM_THREADS) shared(gamma, rho, delta, numObjs)
    {
        int tid = omp_get_thread_num();
        for (int i = tid; i < numObjs; i += NUM_THREADS) {
            gamma[i] = rho[i] * delta[i];
        }
    }

    gamma_index = argsort(numObjs, gamma);
}

void DensityPeaksClustering::fit() {
    get_distance();
    get_rho();   
    get_delta();
    get_gamma();
    int current_category = 1;
    
    for (int seq = 0; seq < numObjs; seq++) {
        int i = gamma_index[seq];
        if (seq >= numClusters)
            break;
        category[i] = current_category;
        center[i] = current_category;
        center_index[seq] = i;
        for (int j = 0; j < numCoords; j++) {
            clusters[seq][j] = objects[i][j];
        }
        current_category++;
    }

    // 因为要按密度从大到小赋予类别，因此这里不能并行
    for (int seq = 0; seq < numObjs; seq ++) {
        int i = rho_index_sorted[seq];
        if (category[i] == 0)
            category[i] = category[nearest[i]];
    }
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

    printf("reading datasets from file %s\n", filename);

    float **objects = file_read(filename, &numObjs, &numCoords);

    DensityPeaksClustering dpc(numObjs, numCoords, numClusters, objects, d_c);

    double start = omp_get_wtime();
    dpc.fit();
    double end = omp_get_wtime();

    printf("\n===== DensityPeaksClustering (OpenMP version) =====\n");

    printf("Input file:       %s\n", filename);
    printf("numObjs         = %d\n", dpc.numObjs);
    printf("numCoords       = %d\n", dpc.numCoords);
    printf("numClusters     = %d\n", dpc.numClusters);
    printf("threshold d_c   = %.4f\n", dpc.d_c);
    printf("clustering time = %.4f milliseconds\n", (end - start) * 1000.0);
    printf("\n===== clustering centers =====\n");
    for (int i = 0; i < numClusters; i++) {
        printf("clustering centers %d: ", i);
        for (int j = 0; j < numCoords; j++) {
            printf("%f ", dpc.clusters[i][j]);
        }
        printf("\n");
    }

    /* output: the coordinates of the cluster centres */
    file_write(filename, numClusters, numObjs, numCoords, dpc.clusters, dpc.category);

    return 0;
}