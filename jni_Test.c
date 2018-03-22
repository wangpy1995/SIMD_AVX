//
// Created by wpy on 18-3-23.
//

#include <jni_Test.h>
#include <immintrin.h>
#include <avx2intrin.h>
#include <avxintrin.h>

JNIEXPORT jintArray JNICALL Java_jni_JniDot_dot_1product_1r7
        (JNIEnv *env, jobject ignore, jint len, jobject src, jobject des0, jobject des1, jobject des2, jobject des3,
         jobject des4,
         jobject des5) {

    int res[6] = {0};

    char *a = (*env)->GetDirectBufferAddress(env, src);
    char *b0 = (*env)->GetDirectBufferAddress(env, des0);
    char *b1 = (*env)->GetDirectBufferAddress(env, des1);
    char *b2 = (*env)->GetDirectBufferAddress(env, des2);
    char *b3 = (*env)->GetDirectBufferAddress(env, des3);
    char *b4 = (*env)->GetDirectBufferAddress(env, des4);
    char *b5 = (*env)->GetDirectBufferAddress(env, des5);
//    char *b6 = (*env)->GetDirectBufferAddress(env, des6);

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

    jintArray jres = (*env)->NewIntArray(env, 6);
    (*env)->SetIntArrayRegion(env, jres, 0, 6, res);
    return jres;
}


JNIEXPORT jint JNICALL Java_jni_JniDot_dot_1product
        (JNIEnv *env, jobject ignore, jobject src, jobject des, jint len) {

    char *a = (*env)->GetDirectBufferAddress(env, src);
    char *b = (*env)->GetDirectBufferAddress(env, des);

    __m256i sum = {0};
    for (int i = 0; i < len; i += 16) {
        __m256i za = _mm256_cvtepi8_epi16(*(__m128i *) (a + i));
        __m256i zb = _mm256_cvtepi8_epi16(*(__m128i *) (b + i));
        sum = _mm256_add_epi32(sum, _mm256_madd_epi16(za, zb));
    }

    int *r = (int *) &sum;
    return r[0] + r[1] + r[2] + r[3] + r[4] + r[5] + r[6] + r[7];
}