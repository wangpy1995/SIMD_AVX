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

void
dot256(float *p, char *a, char *b0, char *b1, char *b2, char *b3, char *b4, char *b5, char *b6, char *b7) {
//    float p[8] = {0.0};
    __m256 sum;
    float *f = (float *) &sum;
    for (int i = 0; i < 512; i += 8) {
//        __m256 za = _mm256_load_ps(a + i);
        /*__m128i *tm = (__m128i *) (a + i);
        __m128i tx = _mm_loadl_epi64(tm);
        __m256 tz = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(tx));
        float *z = (float *) &tz;
        float zz = z[0];*/
        __m256 za = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *) (a + i))));
        __m256 zb0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *) (b0 + i))));
        __m256 zb1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *) (b1 + i))));
        __m256 zb2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *) (b2 + i))));
        __m256 zb3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *) (b3 + i))));
        __m256 zb4 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *) (b4 + i))));
        __m256 zb5 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *) (b5 + i))));
        __m256 zb6 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *) (b6 + i))));
        __m256 zb7 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *) (b7 + i))));
        sum = _mm256_dp_ps(za, zb0, 0xf1);
        p[0] += f[0] + f[4];
        sum = _mm256_dp_ps(za, zb1, 0xf1);
        p[1] += f[0] + f[4];
        sum = _mm256_dp_ps(za, zb2, 0xf1);
        p[2] += f[0] + f[4];
        sum = _mm256_dp_ps(za, zb3, 0xf1);
        p[3] += f[0] + f[4];
        sum = _mm256_dp_ps(za, zb4, 0xf1);
        p[4] += f[0] + f[4];
        sum = _mm256_dp_ps(za, zb5, 0xf1);
        p[5] += f[0] + f[4];
        sum = _mm256_dp_ps(za, zb6, 0xf1);
        p[6] += f[0] + f[4];
        sum = _mm256_dp_ps(za, zb7, 0xf1);
        p[7] += f[0] + f[4];
    }
//    return p[0] + p[1] + p[2] + p[3] + p[4] + p[5] + p[6] + p[7];
//    float *res = (float *) &p;
//    _mm256_store_epi64(p, sum);
//    return (int) (res[0] + res[1] + res[2] + res[3] + res[4] + res[5] + res[6] + res[7]);
}

/*int asm_dot(char *a, char *b, int count) {
    int sum = 0;
    for (int i = 0; i < count; i++) {
        int x = a[i];
        int y = b[i];
        __asm{
        movaps ymm0,[x];
        movaps ymm1,[y];
        mulps ymm0, ymm1;
        }
        }
        return sum;
    }*/

int dot256_epi16(char *a, char *b) {
    bool same = true;
    __m256i tmp_sim;
    int64_t *sim = &tmp_sim;
    //check
    for (int i = 0; i < 512; i += 256) {
        tmp_sim = _mm256_xor_si256(*(__m256i *) (a + i), *(__m256i *) (b + i));
        if (sim[0] != 0 || sim[1] != 0) {
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

void dot128(float *p, char *a, char *b, char *c, char *d, char *e) {
    __m128 sum;
    float *r = &sum;
    for (int i = 0; i < 512; i += 4) {
        __m128 za = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *) (a + i))));
        __m128 zb0 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *) (b + i))));
        __m128 zb1 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *) (c + i))));
        __m128 zb2 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *) (d + i))));
        __m128 zb3 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *) (e + i))));
        sum = _mm_dp_ps(za, zb0, 0xfe);
        p[0] += r[0] + r[4];
        sum = _mm_dp_ps(za, zb1, 0xfe);
        p[1] += r[0] + r[4];
        sum = _mm_dp_ps(za, zb2, 0xfe);
        p[2] += r[0] + r[4];
        sum = _mm_dp_ps(za, zb3, 0xfe);
        p[3] += r[0] + r[4];
    }
}

int dot2(char *a, char *b) {
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

/*float dot128(char *a,char *b){
    __m128i sum;
    float *r = &sum;
    _mm256_load_
}*/


int main(void) {
    /*__m256i sum = {12};
    int32_t *p = (int *) &sum;

    printf("%d\t%d\t%d\t%d\n%d\n%d\n%d\n%d\n", p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
    exit(0);*/
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
        /*float p[8] = {0.0};
        dot256(p, tmp, tmp, tmp, tmp, tmp, tmp, tmp, tmp, tmp);*/
//        float p[4] = {0.0};
//        dot128(p, tmp, tmp, tmp, tmp, tmp);
        sseSum = dot2(data2, data2);
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