#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define INF 1073741824
#define BLOCK_SZ 16

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
    for (int i = 0; i < m * n; i++) {
        fscanf(file, "%d", data + i);
    }

    fclose(file);
    return data;
}

__global__ void distances(int *data, int *dis, int m, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i > j || i >= m || j >= m) return;

    if (i == j) {
        dis[i * m + i] = INF;
    } else {
        int tmp1;
        int tmp2 = 0;
        for (int l = 0; l < n; l++) { // for each dimension
            tmp1 = data[i * n + l] - data[j * n + l];
            tmp2 += tmp1 * tmp1;
        }
        dis[i * m + j] = dis[j * m + i] = tmp2;
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

void knn(int *data, int *result)
{
    int *d_data, *d_result, *d_dis;
    int block = ceil(m / (double)BLOCK_SZ);
    float timer;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMalloc((void**)&d_data, sizeof(int) * m * n);
    cudaMalloc((void**)&d_result, sizeof(int) * m * k);
    cudaMalloc((void**)&d_dis, sizeof(int) * m * m);
    cudaMemcpy(d_data, data, sizeof(int) * m * n, cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    distances<<<dim3(block, block, 1), dim3(BLOCK_SZ, BLOCK_SZ, 1)>>>(d_data, d_dis, m, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timer1, start, stop);

    cudaEventRecord(start);
    sort<<<block2, BLOCK_SZ>>>(d_dis, d_result, m, k);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timer, start, stop);

    cudaMemcpy(result, d_result, sizeof(int) * m * k, cudaMemcpyDeviceToHost);

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
        for (int j = 0; j < k; j++) {
            printf("%d ", result[i * k + j]);
        }
        printf("\n");
    }

    free(data);
    free(result);

    return 0;
}
