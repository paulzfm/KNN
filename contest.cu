#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define INF 1073741824
#define BLOCK_SZ 16

// small
#define SMALL_M 1024
#define SMALL_N 256
#define SMALL_K 8
#define SMALL_B 64

// middle
#define MIDDLE_M 4096
#define MIDDLE_N 1024
#define MIDDLE_K 16
#define MIDDLE_B 256

// large
#define LARGE_M 16384
#define LARGE_N 4096
#define LARGE_K 32
#define LARGE_B 1024

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

__global__ void distances_small(int *data, int *dis)
{
    int tx = threadIdx.x + (threadIdx.z / 2) * BLOCK_SZ / 2;
    int ty = threadIdx.y + (threadIdx.z % 2) * BLOCK_SZ / 2;
    int i = BLOCK_SZ * (blockIdx.x + (blockIdx.z / 2) * SMALL_B / 2) + tx;
    int j = BLOCK_SZ * (blockIdx.y + (blockIdx.z % 2) * SMALL_B / 2) + ty;

    __shared__ int matA[BLOCK_SZ][BLOCK_SZ];
    __shared__ int matB[BLOCK_SZ][BLOCK_SZ];
    int tmp1;
    int tmp2 = 0;

    for (int k = 0; k < SMALL_N; k += BLOCK_SZ) {
        // load sub matrix to shared memory
        matA[tx][ty] = data[i * SMALL_N + (k + ty)];
        matB[tx][ty] = data[j * SMALL_N + (k + tx)];
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
        dis[i * SMALL_M + j] = dis[j * SMALL_M + i] = tmp2;
    } else if (i == j) {
        dis[i * SMALL_M + j] = INF;
    }
}

__global__ void distances_middle(int *data, int *dis)
{
    int tx = threadIdx.x + (threadIdx.z / 2) * BLOCK_SZ / 2;
    int ty = threadIdx.y + (threadIdx.z % 2) * BLOCK_SZ / 2;
    int i = BLOCK_SZ * (blockIdx.x + (blockIdx.z / 2) * MIDDLE_B / 2) + tx;
    int j = BLOCK_SZ * (blockIdx.y + (blockIdx.z % 2) * MIDDLE_B / 2) + ty;

    __shared__ int matA[BLOCK_SZ][BLOCK_SZ];
    __shared__ int matB[BLOCK_SZ][BLOCK_SZ];
    int tmp1;
    int tmp2 = 0;

    for (int k = 0; k < MIDDLE_N; k += BLOCK_SZ) {
        // load sub matrix to shared memory
        matA[tx][ty] = data[i * MIDDLE_N + (k + ty)];
        matB[tx][ty] = data[j * MIDDLE_N + (k + tx)];
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
        dis[i * MIDDLE_M + j] = dis[j * MIDDLE_M + i] = tmp2;
    } else if (i == j) {
        dis[i * MIDDLE_M + j] = INF;
    }
}

__global__ void distances_large(int *data, int *dis)
{
    int tx = threadIdx.x + (threadIdx.z / 2) * BLOCK_SZ / 2;
    int ty = threadIdx.y + (threadIdx.z % 2) * BLOCK_SZ / 2;
    int i = BLOCK_SZ * (blockIdx.x + (blockIdx.z / 2) * LARGE_B / 2) + tx;
    int j = BLOCK_SZ * (blockIdx.y + (blockIdx.z % 2) * LARGE_B / 2) + ty;

    __shared__ int matA[BLOCK_SZ][BLOCK_SZ];
    __shared__ int matB[BLOCK_SZ][BLOCK_SZ];
    int tmp1;
    int tmp2 = 0;

    for (int k = 0; k < LARGE_N; k += BLOCK_SZ) {
        // load sub matrix to shared memory
        matA[tx][ty] = data[i * LARGE_N + (k + ty)];
        matB[tx][ty] = data[j * LARGE_N + (k + tx)];
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
        dis[i * LARGE_M + j] = dis[j * LARGE_M + i] = tmp2;
    } else if (i == j) {
        dis[i * LARGE_M + j] = INF;
    }
}

__global__ void sort_small(int *dis, int *result)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int tmp, idx;
    for (int j = 0; j < SMALL_K; j++) { // find j-th nearest neighbor
        tmp = INF;
        for (int l = i * SMALL_M; l < (i + 1) * SMALL_M; l++) {
            if (dis[l] < tmp) {
                tmp = dis[l];
                idx = l;
            }
        }
        result[i * SMALL_K + j] = idx % SMALL_M;
        dis[idx] = INF;
    }
}

__global__ void sort_middle(int *dis, int *result)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int tmp, idx;
    for (int j = 0; j < MIDDLE_K; j++) { // find j-th nearest neighbor
        tmp = INF;
        for (int l = i * MIDDLE_M; l < (i + 1) * MIDDLE_M; l++) {
            if (dis[l] < tmp) {
                tmp = dis[l];
                idx = l;
            }
        }
        result[i * MIDDLE_K + j] = idx % MIDDLE_M;
        dis[idx] = INF;
    }
}

__global__ void sort_large(int *dis, int *result)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    const int start = i * LARGE_M;
    int buffer[LARGE_K];

    // find the max value in first k elements
    int max = 0;
    int idx;
    for (int j = 0; j < LARGE_K; j++) {
        buffer[j] = j;
        if (dis[start + j] > max) {
            max = dis[start + j];
            idx = j;
        }
    }

    // traverse the remaining elements to select the k minimal
    for (int j = LARGE_K; j < LARGE_M; j++) {
        if (dis[start + j] < max) {
            dis[start + idx] = dis[start + j];
            buffer[idx] = j;
            max = 0;
            for (int l = 0; l < LARGE_K; l++) {
                if (dis[start + l] > max) {
                    max = dis[start + l];
                    idx = l;
                }
            }
        }
    }

    // sort the k elements
    for (int j = 0; j < LARGE_K; j++) { // find j-th nearest neighbor
        max = INF; // use max as "min" here to save register resource
        for (int l = 0; l < LARGE_K; l++) {
            if (dis[start + l] < max ||
                (dis[start + l] == max && buffer[l] < buffer[idx])) {
                max = dis[start + l];
                idx = l;
            }
        }
        result[i * LARGE_K + j] = buffer[idx];
        dis[start + idx] = INF;
    }
}

void knn_small(int *data, int *result, float *timer)
{
    int *d_data, *d_result, *d_dis;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cudaMalloc((void**)&d_data, sizeof(int) * m * n);
    cudaMalloc((void**)&d_result, sizeof(int) * m * k);
    cudaMalloc((void**)&d_dis, sizeof(int) * m * m);
    cudaMemcpy(d_data, data, sizeof(int) * m * n, cudaMemcpyHostToDevice);

    distances_small<<<dim3(SMALL_B / 2, SMALL_B / 2, 4), dim3(BLOCK_SZ / 2, BLOCK_SZ / 2, 4)>>>(d_data, d_dis);
    cudaStreamSynchronize(0);
    sort_small<<<SMALL_B, BLOCK_SZ>>>(d_dis, d_result);
    cudaMemcpy(result, d_result, sizeof(int) * m * k, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(timer, start, stop);
}

void knn_middle(int *data, int *result, float *timer)
{
    int *d_data, *d_result, *d_dis;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cudaMalloc((void**)&d_data, sizeof(int) * m * n);
    cudaMalloc((void**)&d_result, sizeof(int) * m * k);
    cudaMalloc((void**)&d_dis, sizeof(int) * m * m);
    cudaMemcpy(d_data, data, sizeof(int) * m * n, cudaMemcpyHostToDevice);

    distances_middle<<<dim3(MIDDLE_B / 2, MIDDLE_B / 2, 4), dim3(BLOCK_SZ / 2, BLOCK_SZ / 2, 4)>>>(d_data, d_dis);
    cudaStreamSynchronize(0);
    sort_middle<<<MIDDLE_B, BLOCK_SZ>>>(d_dis, d_result);
    cudaMemcpy(result, d_result, sizeof(int) * m * k, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(timer, start, stop);
}

void knn_large(int *data, int *result, float *timer)
{
    int *d_data, *d_result, *d_dis;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cudaMalloc((void**)&d_data, sizeof(int) * m * n);
    cudaMalloc((void**)&d_result, sizeof(int) * m * k);
    cudaMalloc((void**)&d_dis, sizeof(int) * m * m);
    cudaMemcpy(d_data, data, sizeof(int) * m * n, cudaMemcpyHostToDevice);

    distances_large<<<dim3(LARGE_B / 2, LARGE_B / 2, 4), dim3(BLOCK_SZ / 2, BLOCK_SZ / 2, 4)>>>(d_data, d_dis);
    cudaStreamSynchronize(0);
    sort_large<<<LARGE_B, BLOCK_SZ>>>(d_dis, d_result);
    cudaMemcpy(result, d_result, sizeof(int) * m * k, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(timer, start, stop);
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
    float timer;

    if (m == SMALL_M) {
        knn_small(data, result, &timer);
    } else if (m == MIDDLE_M) {
        knn_middle(data, result, &timer);
    } else if (m == LARGE_M) {
        knn_large(data, result, &timer);
    } else {
        fprintf(stderr, "unsupported m: %d\n", m);
        exit(1);
    }

    // output
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            printf("%d ", result[i * k + j]);
        }
        printf("\n");
    }

    if (m == SMALL_M) {
        printf("SMALL:%f\n", timer);
    } else if (m == MIDDLE_M) {
        printf("MIDDLE:%f\n", timer);
    } else if (m == LARGE_M) {
        printf("LARGE:%f\n", timer);
    }

    free(data);
    free(result);

    return 0;
}
