#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define INF (1 << 30)

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

// sequential brute-force method
void knn(int *data, int *dis, int *result)
{
    int i, j, l, tmp, idx;

    // compute distances
    for (i = 0; i < m - 1; i++) {
        for (j = i + 1; j < m; j++) { // for each pair
            tmp = 0;
            for (l = 0; l < n; l++) { // for each dimension
                tmp += (data[i * n + l] - data[j * n + l])
                    * (data[i * n + l] - data[j * n + l]);
            }
            dis[i * m + j] = dis[j * m + i] = tmp;
        }
    }

    for (i = 0; i < m; i++) {
        dis[i * m + i] = INF;
    }

    // find the nearest neighbor
    for (i = 0; i < m; i++) { // for each node
        for (j = 0; j < k; j++) { // find j-th nearest neighbor
            tmp = INF;
            for (l = i * m; l < (i + 1) * m; l++) {
                if (dis[l] < tmp) {
                    tmp = dis[l];
                    idx = l;
                }
            }
            result[i * k + j] = idx % m;
            dis[idx] = INF;
        }
    }
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s input_file\n", argv[0]);
        exit(1);
    }

    int *data = load(argv[1]);
    int *result = (int*)malloc(sizeof(int) * m * k);
    int *dis = (int*)malloc(sizeof(int) * m * m);
    printf("load sample: %d %d %d\n", m, n, k);

    clock_t start, stop;
    start = clock();

    knn(data, dis, result);

    stop = clock();
    printf("time elapsed: %.4lf ms\n", 1000.0 * (stop - start) / CLOCKS_PER_SEC);

    int i, j;
    for (i = 0; i < m; i++) {
        for (j = 0; j < k - 1; j++) {
            printf("%d ", result[i * k + j]);
        }
        printf("%d\n", result[i * k + k - 1]);
    }

    free(data);
    free(result);

    return 0;
}
