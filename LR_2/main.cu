#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <math.h>
#include "device_launch_parameters.h"

#define BlockSize 32

//заполняем матрицу случайными числами
void SetMatrix(double* Mat, int N) {
    for (int i = 0; i < (N * N); i++)
        Mat[i] = rand() % 10;
}

//Прибавляем к i строке j
__global__ void StringAdd(double* M, int i, int j, int N) {
    int ind = blockDim.x * blockIdx.x + threadIdx.x;
    if (ind < N)
        M[i * N + ind] += M[j * N + ind];
}

__global__ void TriangularView(double* M, int N, int i) {
    int thread = blockDim.x * blockIdx.x + threadIdx.x;
    int NumberCurrentRow = thread / N;  //номер текущего ряда
    int NumberCurrentElement = thread % N;  //номер текущего элемента
    double coef = M[N * NumberCurrentRow + i] / M[N * i + i];  //расчет коэффициента
    if (fabs(coef) < 0)
        return;
    __syncthreads();  //ждем пока все треды не достигнут этой точки
    if (i < NumberCurrentRow)
        M[N * NumberCurrentRow + NumberCurrentElement] -= coef * M[N * i + NumberCurrentElement];
    return;
}

//замена нуля
void ReplaceZero(double* M, int N, int i, int blocks) {
    int row = i;  //строка
    if (M[N * row + row] < 0) {
        for (; row < N; row++)
            if (M[N * row + i] > 0)
                break;
    }
    else  // 0 нет
        return;
    if (row >= N)  //вышли за пределы строки
        return;
    StringAdd << < blocks, BlockSize >> > (M, i, row, N);  //запуск ядра StringAdd
    cudaDeviceSynchronize();  //ждем завершения все потоков
}

//подсчет определителя
double Opred(double* M, int N) {
    double Opr = 1;  //изначально определитель =1
    int blocks;
    for (int i = 0; i < N; i++) {
        blocks = (N * N) / BlockSize + 1;
        ReplaceZero(M, N, i, blocks);  //проверка на 0
        TriangularView << <blocks, BlockSize >> > (M, N, i);  //запуск ядра TriangularView
        cudaDeviceSynchronize();  //ждем завершения все потоков
    }
    for (int i = 0; i < N; i++)  //подсчет определителя
        Opr *= M[i * N + i];
    return Opr;  //возвращаем определитель
}

int main() {
    srand(time(NULL));  //установка времени отсчета
    FILE* f = fopen("out.txt", "w");  //открываем файл для записи
    int N = 2, NN = 1002;
    int k = (NN - N) / 10 + 1;  //количество элементов выборки
    double* times = (double*)malloc(k * sizeof(double));  //выделяем память под К элементов
    for (int n = N, i = 0; n <= NN; n += 10) {  //Выборка
        double* a;
        cudaMallocManaged(&a, n * n * sizeof(double));  //выделяем память
        SetMatrix(a, n);  //заполняем матрицу случайными элементами
        clock_t begin = clock();  //время начала
        double Opr = Opred(a, n);  //подсчет определителя
        clock_t end = clock();  //время завершения
        times[i++] = (double)(end - begin) / CLOCKS_PER_SEC;  //записывае затраченное время в массив
        cudaFree(a);  //очищаем память
    }
    for (int n = N, i = 0; n <= NN; n += 10, i++)  //записываем полученные результаты в файл
        fprintf(f, "%d\n%f\n", n, times[i]);
    fclose(f);  //закрываем файл
    return 0;
}
