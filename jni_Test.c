//
// Created by wpy on 18-3-23.
//

#include <jni_Test.h>
#include <immintrin.h>

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

JNIEXPORT jint JNICALL Java_jni_JniDot_dotProduct
        (JNIEnv *env, jobject ignore, jobject src, jobject des) {
    const char *a = (*env)->GetDirectBufferAddress(env, src);
    const char *b = (*env)->GetDirectBufferAddress(env, des);

    __m256i sum = _mm256_setzero_si256();
    int *r = (int *) &sum;

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

    return r[0] + r[1] + r[2] + r[3] + r[4] + r[5] + r[6] + r[7];
}