//
// Created by wpy on 18-4-13.
//
#include <immintrin.h>

#ifndef JNI_TEST_INLINE_AVX_SUM_H
#define JNI_TEST_INLINE_AVX_SUM_H
#endif //JNI_TEST_INLINE_AVX_SUM_H

#define init __m256i za, zb, za0, za1

#define sum256(a, b, i, sum) za = _mm256_load_si256((const __m256i *) ((a) + (i)));\
zb = _mm256_cvtepi8_epi16(*(__m128i *) ((b) + (i)));\
za0 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(za, 0x0));\
za0 = _mm256_madd_epi16(za0, zb);\
zb = _mm256_cvtepi8_epi16(*(__m128i *) ((b) + (i) + 16));\
za1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(za, 0x1));\
za1 = _mm256_madd_epi16(za1, zb);\
za = _mm256_add_epi32(za1, za0);\
(sum) = _mm256_add_epi32((sum), za)