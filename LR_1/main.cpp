#include <stdio.h>
#include <locale.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>

using namespace std;
//прототип функции
double opred(int);

int main() {
    system("chcp 65001");
    FILE* f = fopen("out.txt", "w");
    for (int i = 0, num = 2; num < 1003; num += 10, i++) {
        fprintf(f, "%d\n%f\n", num, opred(num));
    }
    fclose(f);
    printf("Все результаты находятся в документе out.txt");
    return 0;
}

double opred(int n) {  //работа с матрицей
    //выделяем память
    float** matrix = (float**)malloc(n * sizeof(float*));
    for (int i = 0; i < n; i++) {
        matrix[i] = (float*)malloc(n * sizeof(float));
    }
    //Задаем массив рандомными числами
    srand(time(NULL));  //установка времени отсчета
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i][j] = rand() % 10;
        }
    }
    //начало отсчета времени
    double time_start = clock();
    int key = 1;
    for (int i = 0; i < n; ++i) {
        //находим индекс строки с самым большим числом по модулю
        int iMax = i;
        for (int j = i + 1; j < n; ++j)
            if (fabs(matrix[j][i]) > fabs(matrix[iMax][i]))
                iMax = j;
        //ставим это число в начало
        for (int k = 0; k < n; ++k) {
            float q = matrix[i][k];
            matrix[i][k] = matrix[iMax][k];
            matrix[iMax][k] = q;
        }
        //определение знака
        key = -key * (i != iMax ? 1 : -1);
        //приводим к диагональному виду
        for (int j = i + 1; j < n; ++j) {
            float q = -matrix[j][i] / matrix[i][i];
            for (int k = n - 1; k >= i; --k)
                matrix[j][k] += q * matrix[i][k];
        }
    }
    //подсчет определителя
    float opr = 1.0;
    for (int kk = 0; kk < n; kk++) {
        opr = opr * matrix[kk][kk];
    }
    opr *= key;
    //затраченное время
    double time_end = clock();
    double TIME = (double)(time_end - time_start) / CLOCKS_PER_SEC;
    //Очищаем память
    for (int i = 0; i < n; i++)
        free(matrix[i]);
    free(matrix);
    matrix = NULL;
    //возвращаем затраченное время(в секундах)
    return TIME;
}
