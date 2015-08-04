#include <stdio.h>
#include <stdlib.h>

#define INF (1 << 30)
#define BLOCK_SZ 32

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

__global__ void distances(int *data, int *dis, int m, int n)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= m * m) return;
    int i = idx / m;
    int j = idx % m;
    int tmp = 0;
    if (i < j) {
        for (int l = 0; l < n; l++) {
            tmp += (data[i * n + l] - data[j * n + l])
                * (data[i * n + l] - data[j * n + l]);
        }
        dis[i * m + j] = dis[j * m + i] = tmp;
    } else if (i == j) {
        dis[i * m + j] = INF;
    }
}

__global__ void sort(int *dis, int *result, int m, int k)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= m) return;
    int tmp;
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

void knn(int *data, int *result)
{
    int *d_data, *d_result, *d_dis;
    int block1 = ceil(m * m / BLOCK_SZ);
    int block2 = ceil(m / BLOCK_SZ);
    float timer;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cudaMalloc((void**)&d_data, sizeof(int) * m * n);
    cudaMalloc((void**)&d_result, sizeof(int) * m * k);
    cudaMalloc((void**)&d_dis, sizeof(int) * m * m);
    cudaMemcpy(d_data, data, sizeof(int) * m * n, cudaMemcpyHostToDevice);

    distances<<<block1, BLOCK_SZ>>>(d_data, d_dis, m, n);
    cudaStreamSynchronize(0);
    sort<<<block2, BLOCK_SZ>>>(d_dis, d_result, m, k);

    cudaMemcpy(result, d_result, sizeof(int) * m * k, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchorize(stop);
    cudaEventElapsedTime(&timer, start, stop);

    printf("Time elapsed: %.4lf ms\n", timer);

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
