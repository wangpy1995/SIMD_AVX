//
// Created by wangpengyu6 on 18-3-1.
//

#include<stdint.h>
#include <stdio.h>
#include <bits/time.h>
#include <time.h>
#include <immintrin.h>
#include <avxintrin.h>
#include <mhash.h>

/* Defines */
#define ALIGNTO(n) __attribute__((aligned(n)))
#define NUM_ITERS  2560000

int dot256_epi16(char *a, char *b) {
    bool same = true;
    __m256i tmp_sim;
    int64_t *sim = (int64_t *) &tmp_sim;
    //check
    for (int i = 0; i < 512; i += 256) {
        tmp_sim = _mm256_xor_si256(*(__m256i *) (a + i), *(__m256i *) (b + i));
        if (sim[0] == 0 || sim[1] == 0) {
            same = false;
            break;
        }
    }

    //calculate dot_product
    if (same) {
        return 0;
    } else {
        __m256i sum = {0};
        int *p = (int *) &sum;
        for (int i = 0; i < 512; i += 16) {
            __m256i za = _mm256_cvtepi8_epi16(*(__m128i *) (a + i));
            __m256i zb = _mm256_cvtepi8_epi16(*(__m128i *) (b + i));
            sum = _mm256_add_epi32(sum, _mm256_madd_epi16(za, zb));
        }
        return p[0] + p[1] + p[2] + p[3] + p[4] + p[5] + p[6] + p[7];
    }
}

int dot_product(char *a, char *b) {
    bool same = true;
    for (int i = 0; i < 512; i++) {
        if (a[i] != b[i]) {
            same = false;
            break;
        }
    }
    if (same) {
        return 0;
    } else {
        int sum = 0;
        for (int i = 0; i < 512; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }
}


int main(void) {
    /* Variables */
    /* Loop Counter */
    /* Data to process */
    char data[512] ALIGNTO(32);
    char data2[512] ALIGNTO(32);
//    unsigned char mask[16]  ALIGNTO(16);
    int sseSum = 0;
    int sseSum2 = 0;

    /* Time tracking */
    clock_t t1, t2, t3;
    double sseTime, sseTime2;



    /* Initialize mask and float arrays with some random data. */
    for (int i = -127; i < 384; i++) {
        data[i + 127] = (char) (i % 128);
        data2[i + 127] = (char) (i % 128);
//        data2[i + 127] = (char) 127;
    }


    /* RUN TESTS */

    char tmp[512] ALIGNTO(32);
    for (int i = 0; i < 512; i++) {
        tmp[i] = data[i];
    }
    t1 = clock();

    for (int i = 0; i < NUM_ITERS; i++) {
//        float p[4] = {0.0};
//        dot128(p, tmp, tmp, tmp, tmp, tmp);
        sseSum = dot_product(data2, data2);
    }
    t2 = clock();
    for (int i = 0; i < NUM_ITERS; i++) {
        sseSum2 = dot256_epi16(data2, data2);
    }
    t3 = clock();


    /* Compute time taken */
    sseTime = (double) (t2 - t1) / CLOCKS_PER_SEC;
    sseTime2 = (double) (t3 - t2) / CLOCKS_PER_SEC;


    /* Print out results */
    printf("Results:\nSSE:  Time: %f    Value: %d\nResults:\nSSE2:  Time: %f    Value: %d\n",
           sseTime, sseSum,
           sseTime2, sseSum2);

    return 0;
}