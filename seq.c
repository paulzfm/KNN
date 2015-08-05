#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define INF 1073741824

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
void knn(int *data, int *result)
{
    int *dis = (int*)malloc(sizeof(int) * m * m);
    int i, j, l, tmp, idx;
    clock_t start, stop;

    start = clock();

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

    stop = clock();
    fprintf(stderr, "distance: %.4lf ms\n", 1000.0 * (stop - start) / CLOCKS_PER_SEC);

    start = clock();

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

    stop = clock();
    fprintf(stderr, "sort: %.4lf ms\n", 1000.0 * (stop - start) / CLOCKS_PER_SEC);

    free(dis);
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s input_file\n", argv[0]);
        exit(1);
    }

    int *data = load(argv[1]);
    int *result = (int*)malloc(sizeof(int) * m * k);

    knn(data, result);

    int i, j;
    for (i = 0; i < m; i++) {
        for (j = 0; j < k; j++) {
            printf("%d ", result[i * k + j]);
        }
        printf("\n");
    }

    free(data);
    free(result);

    return 0;
}
