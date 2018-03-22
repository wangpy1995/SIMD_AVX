//
// Created by wangpengyu6 on 18-3-1.
//

#include<stdint.h>
#include <stdio.h>
#include <bits/time.h>
#include <time.h>
#include <immintrin.h>
#include <avxintrin.h>
#include <stdbool.h>

/* Defines */
#define ALIGNTO(n) __attribute__((aligned(n)))
#define NUM_ITERS  2560000

int dot256_epi16(const char *a, const char *b) {
    bool same = true;
    __m256i tmp_sim;
    int64_t *sim = (int64_t *) &tmp_sim;
    //check
//    for (int i = 0; i < 512; i += 256) {
//        tmp_sim = _mm256_xor_si256(*(__m256i *) (a + i), *(__m256i *) (b + i));
//        if (sim[0] != 0 || sim[1] != 0 || sim[3] != 0 || sim[4] != 0) {
//            same = false;
//            break;
//        }
//    }

    //calculate dot_product
//    if (same) {
//        return 0;
//    } else {
    __m256i sum = {0};
    int *p = (int *) &sum;
    for (int i = 0; i < 512; i += 16) {
        __m256i za = _mm256_cvtepi8_epi16(*(__m128i *) (a + i));
        __m256i zb = _mm256_cvtepi8_epi16(*(__m128i *) (b + i));
        sum = _mm256_add_epi32(sum, _mm256_madd_epi16(za, zb));
    }
    return p[0] + p[1] + p[2] + p[3] + p[4] + p[5] + p[6] + p[7];
//    }
}

void
dot256_epi16_r7(int *res, const char *a, const char *b0, const char *b1, const char *b2, const char *b3, const char *b4,
                const char *b5, const char *b6) {

    __m256i sum0 = {0};
    __m256i sum1 = {0};
    __m256i sum2 = {0};
    __m256i sum3 = {0};
    __m256i sum4 = {0};
    __m256i sum5 = {0};
//    __m256i sum6 = {0};
    for (int i = 0; i < 512; i += 16) {
        __m256i za = _mm256_cvtepi8_epi16(*(__m128i *) (a + i));

        __m256i zb0 = _mm256_cvtepi8_epi16(*(__m128i *) (b0 + i));
        __m256i zb1 = _mm256_cvtepi8_epi16(*(__m128i *) (b1 + i));
        __m256i zb2 = _mm256_cvtepi8_epi16(*(__m128i *) (b2 + i));
        __m256i zb3 = _mm256_cvtepi8_epi16(*(__m128i *) (b3 + i));
        __m256i zb4 = _mm256_cvtepi8_epi16(*(__m128i *) (b4 + i));
        __m256i zb5 = _mm256_cvtepi8_epi16(*(__m128i *) (b5 + i));
//        __m256i zb6 = _mm256_cvtepi8_epi16(*(__m128i *) (b6 + i));

        sum0 = _mm256_add_epi32(sum0, _mm256_madd_epi16(za, zb0));
        sum1 = _mm256_add_epi32(sum1, _mm256_madd_epi16(za, zb1));
        sum2 = _mm256_add_epi32(sum2, _mm256_madd_epi16(za, zb2));
        sum3 = _mm256_add_epi32(sum3, _mm256_madd_epi16(za, zb3));
        sum4 = _mm256_add_epi32(sum4, _mm256_madd_epi16(za, zb4));
        sum5 = _mm256_add_epi32(sum5, _mm256_madd_epi16(za, zb5));
//        sum6 = _mm256_add_epi32(sum6, _mm256_madd_epi16(za, zb6));
    }

    int *r0 = (int *) &sum0;
    int *r1 = (int *) &sum1;
    int *r2 = (int *) &sum2;
    int *r3 = (int *) &sum3;
    int *r4 = (int *) &sum4;
    int *r5 = (int *) &sum5;
//    int *r6 = (int *) &sum6;

    res[0] = r0[0] + r0[1] + r0[2] + r0[3] + r0[4] + r0[5] + r0[6] + r0[7];
    res[1] = r1[0] + r1[1] + r1[2] + r1[3] + r1[4] + r1[5] + r1[6] + r1[7];
    res[2] = r2[0] + r2[1] + r2[2] + r2[3] + r2[4] + r2[5] + r2[6] + r2[7];
    res[3] = r3[0] + r3[1] + r3[2] + r3[3] + r3[4] + r3[5] + r3[6] + r3[7];
    res[4] = r4[0] + r4[1] + r4[2] + r4[3] + r4[4] + r4[5] + r4[6] + r4[7];
    res[5] = r5[0] + r5[1] + r5[2] + r5[3] + r5[4] + r5[5] + r5[6] + r5[7];
//    res[6] = r6[0] + r6[1] + r6[2] + r6[3] + r6[4] + r6[5] + r6[6] + r6[7];
}

int dot_product(const char *a, const char *b) {
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

    int res[6] = {0};
    for (int i = 0; i < NUM_ITERS / 6; i++) {
//        float p[4] = {0.0};
//        dot128(p, tmp, tmp, tmp, tmp, tmp);
        dot256_epi16_r7(res, data, data2, data2, data2, data2, data2, data2, data2);
    }
    sseSum = res[5];
    t2 = clock();
    for (int i = 0; i < NUM_ITERS; i++) {
        sseSum2 = dot256_epi16(data2, data2);
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