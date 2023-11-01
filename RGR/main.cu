#include <iostream>
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <locale.h>
#include <math.h>

#define N 10000 //Количество столбцов/строк матрицы

int A[N * N];
int B[N * N];
int* d_a;
int* d_b;

const int BLOCK_COLS = N < 32 ? N : 32;  // Количество потоков в блоке по оси X
const int BLOCK_ROWS = N < 32 ? N : 32;  // Количество потоков в блоке по оси Y

__global__ void transpose(int* inputMatrix, int* outputMatrix, int width, int height)
{
    int Index_x = blockDim.x * blockIdx.x + threadIdx.x;
    int Index_y = blockDim.y * blockIdx.y + threadIdx.y;

    if ((Index_x < width) && (Index_y < height))
    {
        //Линейный индекс элемента строки исходной матрицы
        int inputIdx = Index_x + width * Index_y;

        //Линейный индекс элемента столбца матрицы-результата
        int outputIdx = Index_y + height * Index_x;

        outputMatrix[outputIdx] = inputMatrix[inputIdx];
    }
}

__global__ void transpose_shared(int *inputMatrix, int *outputMatrix, int width, int height)
{
    __shared__ float temp[BLOCK_ROWS][BLOCK_COLS+1];

    int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

    if ((xIndex < width) && (yIndex < height))
    {
        // Линейный индекс элемента строки исходной матрицы
        int idx = yIndex * width + xIndex;

        //Копируем элементы исходной матрицы
        temp[threadIdx.y][threadIdx.x] = inputMatrix[idx];
    }

    //Синхронизируем все нити в блоке
    __syncthreads();

    xIndex = blockIdx.y * blockDim.y + threadIdx.x;
    yIndex = blockIdx.x * blockDim.x + threadIdx.y;

    if ((xIndex < height) && (yIndex < width))
    {
        // Линейный индекс элемента строки исходной матрицы
        int idx = yIndex * height + xIndex;

        //Копируем элементы исходной матрицы
        outputMatrix[idx] = temp[threadIdx.x][threadIdx.y];
    }
}

void Vivodilo(int* A)
{
    int i, j;
    for (i = 0; i < N; i++)
    {
        printf("|");
        for (j = 0; j < N; j++)
        {
            if (A[i * N + j] >= 0)
                printf(" %d|", A[i * N + j]);
            else
                printf("%d|", A[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main()
{
    system("chcp 65001");

    printf("Выберите режим работы: 28 - Без разделяемой памяти, 36 - С разделяемой памятью\n");
    int mode;
    scanf("%i", &mode);

    printf("Выводить матрицы на экран?: 29 - Да, 37 - Нет\n");
    int vivod;
    scanf("%i", &vivod);

    int i, j;
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
        {
            A[i * N + j] = (rand() % 4);
            //A[i * N + j] = i*j + 1;
        }

    cudaMalloc((void**)&d_a, (N * N) * sizeof(int)); //выделение памяти на device
    cudaMalloc((void**)&d_b, (N * N) * sizeof(int));

    cudaMemcpy(d_a, &A, (N * N) * sizeof(int), cudaMemcpyHostToDevice);

    const int nx = N;
    const int ny = N;

    dim3 dimGrid(nx / BLOCK_COLS, ny / BLOCK_ROWS, 1);
    dim3 dimBlock(BLOCK_COLS, BLOCK_ROWS, 1);

    cudaEvent_t start;
    cudaEvent_t stop;

    //Создаем event для синхронизации и замера времени работы GPU
    (cudaEventCreate(&start));
    (cudaEventCreate(&stop));

    cudaEventRecord(start, 0);

    if (mode == 0)
    {
        transpose <<<dimGrid,dimBlock>>> (d_a, d_b, N, N);
    }
    else
    {
        transpose_shared <<<dimGrid,dimBlock>>> (d_a, d_b, N, N);
    }

    cudaEventRecord(stop, 0);

    float time = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    cudaMemcpy(B, d_b, (N * N) * sizeof(int), cudaMemcpyDeviceToHost);

    if (vivod == 0)
    {
        Vivodilo(A);
        Vivodilo(B);
    }

    cudaFree(d_a);
    cudaFree(d_b);

    printf("Матрица успешно транспонирована\n");
    std::cout << time << std::endl;
    printf("Время выполнения: %.0f мс\n", time);

    system("pause");
    return 0;
}
