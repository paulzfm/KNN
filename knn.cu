#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <thrust/sort.h>

#define INF 1073741824
#define BLOCK_SZ 16
#define HEAP_SZ 32
#define MAX_M 16384

int m; // nodes
int n; // dimensions
int k; // k-nearest

// input sample file
int* load(const char *input)
{
    FILE *file = fopen(input, "r");
    if (!file) {
        fprintf(stderr, "Error: no such input file \"%s\"\n", input);
        exit(1);
    }

    // load m, n, k
    fscanf(file, "%d%d%d", &m, &n, &k);

    // allocate memory
    int *data = (int*)malloc(sizeof(int) * m * n);

    // load data
    int i;
    for (i = 0; i < m * n; i++) {
        fscanf(file, "%d", data + i);
    }

    fclose(file);
    return data;
}

__global__ void distances2(int *data, int *dis, int m, int n)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int i = BLOCK_SZ * blockIdx.x + tx;
    int j = BLOCK_SZ * blockIdx.y + ty;
    if (i >= m || j >= m) return;

    __shared__ int matA[BLOCK_SZ][BLOCK_SZ];
    __shared__ int matB[BLOCK_SZ][BLOCK_SZ];
    int tmp1;
    int tmp2 = 0;

    for (int k = 0; k < n; k += BLOCK_SZ) {
        // load sub matrix to shared memory
        matA[tx][ty] = (k + ty < n) ? data[i * n + (k + ty)] : 0;
        // matB[tx][ty] = (k + tx < n) ? data[j * n + (k + tx)] : 0;
        matB[ty][tx] = (k + tx < n) ? data[j * n + (k + tx)] : 0;
        __syncthreads();

        if (i < j) { // compute partial sum
            for (int w = 0; w < BLOCK_SZ; w++) {
                // tmp1 = matA[tx][w] - matB[w][ty];
                tmp1 = matA[tx][w] - matB[ty][w];
                tmp2 += tmp1 * tmp1;
            }
        }
        __syncthreads();
    }

    if (i < j) {
        dis[i * m + j] = dis[j * m + i] = tmp2;
    } else if (i == j) {
        dis[i * m + j] = INF;
    }
}

__global__ void distances3(int *data, int *dis, int m, int n, int block)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int i = BLOCK_SZ * (blockIdx.x + (blockIdx.z / 2) * block) + tx;
    int j = BLOCK_SZ * (blockIdx.y + (blockIdx.z % 2) * block) + ty;
    if (i >= m || j >= m) return;

    __shared__ int matA[BLOCK_SZ][BLOCK_SZ];
    __shared__ int matB[BLOCK_SZ][BLOCK_SZ];
    int tmp1;
    int tmp2 = 0;

    for (int k = 0; k < n; k += BLOCK_SZ) {
        // load sub matrix to shared memory
        matA[tx][ty] = (k + ty < n) ? data[i * n + (k + ty)] : 0;
        matB[tx][ty] = (k + tx < n) ? data[j * n + (k + tx)] : 0;
        __syncthreads();

        if (i < j) { // compute partial sum
            for (int w = 0; w < BLOCK_SZ; w++) {
                tmp1 = matA[tx][w] - matB[w][ty];
                tmp2 += tmp1 * tmp1;
            }
        }
        __syncthreads();
    }

    if (i < j) {
        dis[i * m + j] = dis[j * m + i] = tmp2;
    } else if (i == j) {
        dis[i * m + j] = INF;
    }
}

__global__ void distances33(int *data, int *dis, int m, int n, int block)
{
    int tx = threadIdx.x + (threadIdx.z / 2) * BLOCK_SZ / 2;
    int ty = threadIdx.y + (threadIdx.z % 2) * BLOCK_SZ / 2;
    int i = BLOCK_SZ * (blockIdx.x + (blockIdx.z / 2) * block) + tx;
    int j = BLOCK_SZ * (blockIdx.y + (blockIdx.z % 2) * block) + ty;
    if (i >= m || j >= m) return;

    __shared__ int matA[BLOCK_SZ][BLOCK_SZ];
    __shared__ int matB[BLOCK_SZ][BLOCK_SZ];
    int tmp1;
    int tmp2 = 0;

    for (int k = 0; k < n; k += BLOCK_SZ) {
        // load sub matrix to shared memory
        matA[tx][ty] = (k + ty < n) ? data[i * n + (k + ty)] : 0;
        matB[tx][ty] = (k + tx < n) ? data[j * n + (k + tx)] : 0;
        __syncthreads();

        if (i < j) { // compute partial sum
            for (int w = 0; w < BLOCK_SZ; w++) {
                tmp1 = matA[tx][w] - matB[w][ty];
                tmp2 += tmp1 * tmp1;
            }
        }
        __syncthreads();
    }

    if (i < j) {
        dis[i * m + j] = dis[j * m + i] = tmp2;
    } else if (i == j) {
        dis[i * m + j] = INF;
    }
}

__global__ void sort(int *dis, int *result, int m, int k)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= m) return;
    int tmp, idx;
    for (int j = 0; j < k; j++) { // find j-th nearest neighbor
        tmp = INF;
        for (int l = i * m; l < (i + 1) * m; l++) {
            if (dis[l] < tmp) {
                tmp = dis[l];
                idx = l;
            }
        }
        result[i * k + j] = idx % m;
        dis[idx] = INF;
    }
}

__device__ void sort_by_key(int *dis, int *result, int i, int m, int k)
{
    int values[MAX_M];
    for (int j = 0; j < m; j++) {
        values[j] = j;
    }
    thrust::sort_by_key(dis + (i * m), dis + (i + 1) * m, values);
    for (int j = 0; j < k; j++) {
        result[i * k + j] = values[j];
    }
}

__global__ void tsort(int *dis, int *result, int m, int k)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= m) return;
    sort_by_key(dis, result, i, m, k);
}

void knn(int *data, int *result)
{
    int *d_data, *d_result, *d_dis;
    int *dis = (int*)malloc(sizeof(int) * m * m);
    int block = ceil(m / (double)BLOCK_SZ);
    int block1 = ceil(block / 2.0);
    float timer1, timer2;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMalloc((void**)&d_data, sizeof(int) * m * n);
    cudaMalloc((void**)&d_result, sizeof(int) * m * k);
    cudaMalloc((void**)&d_dis, sizeof(int) * m * m);
    cudaMemcpy(d_data, data, sizeof(int) * m * n, cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    // distances2<<<dim3(block, block, 1), dim3(BLOCK_SZ, BLOCK_SZ, 1)>>>(d_data, d_dis, m, n);
    // distances3<<<dim3(block1, block1, 4), dim3(BLOCK_SZ, BLOCK_SZ, 1)>>>(d_data, d_dis, m, n, block1);
    distances33<<<dim3(block1, block1, 4), dim3(BLOCK_SZ / 2, BLOCK_SZ / 2, 4)>>>(d_data, d_dis, m, n, block1);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timer1, start, stop);

    cudaEventRecord(start);
    // sort<<<block, BLOCK_SZ>>>(d_dis, d_result, m, k);
    tsort<<<block, BLOCK_SZ>>>(d_dis, d_result, m, k);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timer2, start, stop);

    cudaMemcpy(result, d_result, sizeof(int) * m * k, cudaMemcpyDeviceToHost);

    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&timer, start, stop);

    fprintf(stderr, "distance: %.4lf ms\n", timer1);
    fprintf(stderr, "sort: %.4lf ms\n", timer2);

    cudaFree(d_data);
    cudaFree(d_result);
    cudaFree(d_dis);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s input_file\n", argv[0]);
        exit(1);
    }

    // input
    int *data = load(argv[1]);
    int *result = (int*)malloc(sizeof(int) * m * k);

    // compute
    knn(data, result);

    // output
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k - 1; j++) {
            printf("%d ", result[i * k + j]);
        }
        printf("%d\n", result[i * k + k - 1]);
    }

    free(data);
    free(result);

    return 0;
}
