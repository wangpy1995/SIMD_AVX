//
// Created by wangpengyu6 on 18-3-1.
//

#include <stdio.h>
#include <bits/time.h>
#include <time.h>
#include <immintrin.h>
//#include "inline_avx_sum.h"

/* Defines */
#define ALIGNTO(n) __attribute__((aligned(n)))
#define NUM_ITERS  256000000
#define LEN 512

int dot_product(const char *a, const char *b) {
  // 0
  __m256i za0 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i *) a));
  a += 16;
  __m256i zb0 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i *) b));
  b += 16;
  __m256i sum0 = _mm256_madd_epi16(za0, zb0);
  __m256i za1 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i *) a));
  a += 16;
  __m256i zb1 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i *) b));
  b += 16;
  __m256i sum1 = _mm256_madd_epi16(za1, zb1);

  // 1
  __m256i za2 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i *) a));
  a += 16;
  __m256i zb2 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i *) b));
  b += 16;
  __m256i sum2 = _mm256_madd_epi16(za2, zb2);
  __m256i za3 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i *) a));
  a += 16;
  __m256i zb3 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i *) b));
  b += 16;
  __m256i sum3 = _mm256_madd_epi16(za3, zb3);

  // 2
  __m256i za4 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i *) a));
  a += 16;
  __m256i zb4 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i *) b));
  b += 16;
  __m256i sum4 = _mm256_madd_epi16(za4, zb4);
  __m256i za5 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i *) a));
  a += 16;
  __m256i zb5 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i *) b));
  b += 16;
  __m256i sum5 = _mm256_madd_epi16(za5, zb5);

  // 3
  __m256i za6 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i *) a));
  a += 16;
  __m256i zb6 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i *) b));
  b += 16;
  __m256i sum6 = _mm256_madd_epi16(za6, zb6);
  __m256i za7 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i *) a));
  a += 16;
  __m256i zb7 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i *) b));
  b += 16;
  __m256i sum7 = (_mm256_madd_epi16(za7, zb7));

  // 4
  __m256i za8 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i *) a));
  a += 16;
  __m256i zb8 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i *) b));
  b += 16;
  __m256i sum8 = _mm256_madd_epi16(za8, zb8);
  __m256i za9 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i *) a));
  a += 16;
  __m256i zb9 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i *) b));
  b += 16;
  __m256i sum9 = _mm256_madd_epi16(za9, zb9);

  // 5
  __m256i za10 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i *) a));
  a += 16;
  __m256i zb10 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i *) b));
  b += 16;
  __m256i sum10 = _mm256_madd_epi16(za10, zb10);
  __m256i za11 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i *) a));
  a += 16;
  __m256i zb11 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i *) b));
  b += 16;
  __m256i sum11 = _mm256_madd_epi16(za11, zb11);

  // 6
  __m256i za12 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i *) a));
  a += 16;
  __m256i zb12 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i *) b));
  b += 16;
  __m256i sum12 = _mm256_madd_epi16(za12, zb12);
  __m256i za13 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i *) a));
  a += 16;
  __m256i zb13 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i *) b));
  b += 16;
  __m256i sum13 = _mm256_madd_epi16(za13, zb13);

  // 7
  __m256i za14 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i *) a));
  a += 16;
  __m256i zb14 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i *) b));
  b += 16;
  __m256i sum14 = _mm256_madd_epi16(za14, zb14);
  __m256i za15 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i *) a));
  a += 16;
  __m256i zb15 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i *) b));
  b += 16;
  __m256i sum15 = _mm256_madd_epi16(za15, zb15);

  // 8
  __m256i za16 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i *) a));
  a += 16;
  __m256i zb16 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i *) b));
  b += 16;
  __m256i sum16 = _mm256_madd_epi16(za16, zb16);
  __m256i za17 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i *) a));
  a += 16;
  __m256i zb17 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i *) b));
  b += 16;
  __m256i sum17 = _mm256_madd_epi16(za17, zb17);

  // 9
  __m256i za18 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i *) a));
  a += 16;
  __m256i zb18 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i *) b));
  b += 16;
  __m256i sum18 = _mm256_madd_epi16(za18, zb18);
  __m256i za19 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i *) a));
  a += 16;
  __m256i zb19 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i *) b));
  b += 16;
  __m256i sum19 = _mm256_madd_epi16(za19, zb19);

  // 10
  __m256i za20 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i *) a));
  a += 16;
  __m256i zb20 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i *) b));
  b += 16;
  __m256i sum20 = _mm256_madd_epi16(za20, zb20);
  __m256i za21 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i *) a));
  a += 16;
  __m256i zb21 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i *) b));
  b += 16;
  __m256i sum21 = _mm256_madd_epi16(za21, zb21);

  // 11
  __m256i za22 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i *) a));
  a += 16;
  __m256i zb22 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i *) b));
  b += 16;
  __m256i sum22 = _mm256_madd_epi16(za22, zb22);
  __m256i za23 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i *) a));
  a += 16;
  __m256i zb23 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i *) b));
  b += 16;
  __m256i sum23 = _mm256_madd_epi16(za23, zb23);

  // 12
  __m256i za24 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i *) a));
  a += 16;
  __m256i zb24 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i *) b));
  b += 16;
  __m256i sum24 = _mm256_madd_epi16(za24, zb24);
  __m256i za25 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i *) a));
  a += 16;
  __m256i zb25 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i *) b));
  b += 16;
  __m256i sum25 = _mm256_madd_epi16(za25, zb25);

  // 13
  __m256i za26 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i *) a));
  a += 16;
  __m256i zb26 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i *) b));
  b += 16;
  __m256i sum26 = _mm256_madd_epi16(za26, zb26);
  __m256i za27 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i *) a));
  a += 16;
  __m256i zb27 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i *) b));
  b += 16;
  __m256i sum27 = _mm256_madd_epi16(za27, zb27);

  // 14
  __m256i za28 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i *) a));
  a += 16;
  __m256i zb28 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i *) b));
  b += 16;
  __m256i sum28 = _mm256_madd_epi16(za28, zb28);
  __m256i za29 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i *) a));
  a += 16;
  __m256i zb29 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i *) b));
  b += 16;
  __m256i sum29 = _mm256_madd_epi16(za29, zb29);

  // 15
  __m256i za30 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i *) a));
  a += 16;
  __m256i zb30 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i *) b));
  b += 16;
  __m256i sum30 = _mm256_madd_epi16(za30, zb30);
  __m256i za31 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i *) a));
  __m256i zb31 = _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i *) b));
  __m256i sum31 = _mm256_madd_epi16(za31, zb31);

  __m256i s0 = _mm256_add_epi32(_mm256_add_epi32(sum0, sum1), _mm256_add_epi32(sum2, sum3));
  __m256i s1 = _mm256_add_epi32(_mm256_add_epi32(sum4, sum5), _mm256_add_epi32(sum6, sum7));
  __m256i s2 = _mm256_add_epi32(_mm256_add_epi32(sum8, sum9), _mm256_add_epi32(sum10, sum11));
  __m256i s3 = _mm256_add_epi32(_mm256_add_epi32(sum12, sum13), _mm256_add_epi32(sum14, sum15));
  __m256i s4 = _mm256_add_epi32(_mm256_add_epi32(sum16, sum17), _mm256_add_epi32(sum18, sum19));
  __m256i s5 = _mm256_add_epi32(_mm256_add_epi32(sum20, sum21), _mm256_add_epi32(sum22, sum23));
  __m256i s6 = _mm256_add_epi32(_mm256_add_epi32(sum24, sum25), _mm256_add_epi32(sum26, sum27));
  __m256i s7 = _mm256_add_epi32(_mm256_add_epi32(sum28, sum29), _mm256_add_epi32(sum30, sum31));

  __m256i r0 = _mm256_add_epi32(_mm256_add_epi32(s0, s1), _mm256_add_epi32(s2, s3));
  __m256i r1 = _mm256_add_epi32(_mm256_add_epi32(s4, s5), _mm256_add_epi32(s6, s7));

  __m256i res = _mm256_add_epi32(r0, r1);
  int32_t *p = (int32_t *) &res;
  return p[0] + p[1] + p[2] + p[3] + p[4] + p[5] + p[6] + p[7];
}

int main(void) {

  printf("iteration num: %d\nfeat len: %d\n", NUM_ITERS, LEN);
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
  int i;
  for (i = 0; i < LEN; i++) {
    data[i] = (char) i;
    data2[i] = (char) i;
  }

  int x, y = 0;
  for (x = 0; x < LEN; x++) {
    y += data[x] * data2[x];
  }
  printf("dot prod res: %d\n", y);


  /* RUN TESTS */

//    int iters = rand() % NUM_ITERS;

  t1 = clock();
  for (i = 0; i < NUM_ITERS; i++) {
    data[0] = (char) i;
    sseSum = dot_product(data, data2);
  }
  t2 = clock();
  for (i = 0; i < NUM_ITERS; i++) {
    data[0] = (char) i;
//    sseSum2 = dot256_epi16(data, data2);
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