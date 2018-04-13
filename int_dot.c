//
// Created by wangpengyu6 on 18-3-1.
//

#include <stdio.h>
#include <bits/time.h>
#include <time.h>
#include <inline_avx_sum.h>

/* Defines */
#define ALIGNTO(n) __attribute__((aligned(n)))
#define NUM_ITERS  25600000
#define LEN 512

int dot256_epi16(const char *a, const char *b) {
    __m256i sum = _mm256_setzero_si256();
    int *p = (int *) &sum;
    init;
    sum256(a, b, 32 * 0, sum);
    sum256(a, b, 32 * 1, sum);
    sum256(a, b, 32 * 2, sum);
    sum256(a, b, 32 * 3, sum);

    sum256(a, b, 32 * 4, sum);
    sum256(a, b, 32 * 5, sum);
    sum256(a, b, 32 * 6, sum);
    sum256(a, b, 32 * 7, sum);

    sum256(a, b, 32 * 8, sum);
    sum256(a, b, 32 * 9, sum);
    sum256(a, b, 32 * 10, sum);
    sum256(a, b, 32 * 11, sum);

    sum256(a, b, 32 * 12, sum);
    sum256(a, b, 32 * 13, sum);
    sum256(a, b, 32 * 14, sum);
    sum256(a, b, 32 * 15, sum);

    return p[0] + p[1] + p[2] + p[3] + p[4] + p[5] + p[6] + p[7];
}

int dot_product(const char *a, const char *b) {
    int sum = 0;
    for (int i = 0; i < LEN; i++) {
        int temp = a[i] * b[i];
        sum = sum + temp;
    }
    return sum;
}

int main(void) {
    /* Variables */
    /* Loop Counter */
    /* Data to process */
    char data[512] ALIGNTO(32);
    char data2[512] ALIGNTO(32);
    int sseSum = 0;
    int sseSum2 = 0;

    /* Time tracking */
    clock_t t1, t2, t3;
    double sseTime, sseTime2;



    /* Initialize mask and float arrays with some random data. */
    for (int i = 0; i < LEN; i++) {
        data[i] = (char) i;
        data2[i] = (char) i;
    }


    /* RUN TESTS */

    int iters = rand() % NUM_ITERS;

    t1 = clock();

    for (int i = 0; i < NUM_ITERS; i++) {
        data[0] = (char) i;
        sseSum = dot_product(data, data2);
    }
    t2 = clock();
    for (int i = 0; i < NUM_ITERS; i++) {
        data[0] = (char) i;
        sseSum2 = dot256_epi16(data, data2);
    }
    t3 = clock();


    /* Compute time taken */
    sseTime = (double) (t2 - t1) / CLOCKS_PER_SEC;
    sseTime2 = (double) (t3 - t2) / CLOCKS_PER_SEC;


    /* Print out results */
    printf("Results:\n"
                   "SSE:  Time: %f    Value: %d\nResults:\n"
                   "SSE2:  Time: %f    Value: %d\n",
           sseTime, sseSum,
           sseTime2, sseSum2);

    return 0;
}