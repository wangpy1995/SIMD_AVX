//
// Created by wangpengyu6 on 18-3-2.
//

#include <com_hikvision_algorithm_JNIComparator.h>
#include <immintrin.h>
#include <avxintrin.h>

//内存对齐操作
#define ALIGNTO(n) __attribute__((aligned(n)))
#define LEN 512
//请求比对对象的地址
int i;
int8_t *src_addr;
jint *result;
float src[LEN] ALIGNTO(32);
float dest0[LEN] ALIGNTO(32);
float dest1[LEN] ALIGNTO(32);
float dest2[LEN] ALIGNTO(32);
float dest3[LEN] ALIGNTO(32);
float dest4[LEN] ALIGNTO(32);
float dest5[LEN] ALIGNTO(32);
float dest6[LEN] ALIGNTO(32);
float dest7[LEN] ALIGNTO(32);

JNIEXPORT void JNICALL Java_com_hikvision_algorithm_JNIComparator_updateAddress
        (JNIEnv *env, jobject obj, jobject m1, jobject res) {
    src_addr = (*env)->GetDirectBufferAddress(env, m1);
    result = (*env)->GetDirectBufferAddress(env, res);
    for (i = 0; i < LEN; i++) {
        src[i] = src_addr[i];
    }
}

//float p[8] = {0.0};

void *
dot256(float *p, float *a, float *b0, float *b1, float *b2, float *b3, float *b4, float *b5, float *b6, float *b7) {
    __m256 sum;
    float *f = (float *) &sum;
    for (int i = 0; i < LEN; i += 8) {
        __m256 za = _mm256_load_ps(a + i);
        __m256 zb0 = _mm256_load_ps(b0 + i);
        __m256 zb1 = _mm256_load_ps(b1 + i);
        __m256 zb2 = _mm256_load_ps(b2 + i);
        __m256 zb3 = _mm256_load_ps(b3 + i);
        __m256 zb4 = _mm256_load_ps(b4 + i);
        __m256 zb5 = _mm256_load_ps(b5 + i);
        __m256 zb6 = _mm256_load_ps(b6 + i);
        __m256 zb7 = _mm256_load_ps(b7 + i);
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
    return p;
}
typedef int UINT;
typedef char UCHAR;
UINT EndianConvertLToB(UINT InputNum) {
    UCHAR *p = (UCHAR *) &InputNum;
    return (((UINT) *p << 24) + ((UINT) *(p + 1) << 16) +
            ((UINT) *(p + 2) << 8) + (UINT) *(p + 3));
}

JNIEXPORT void JNICALL Java_com_hikvision_algorithm_JNIComparator_compare
        (JNIEnv *env, jobject obj, jobject m0, jobject m1, jobject m2, jobject m3, jobject m4, jobject m5, jobject m6,
         jobject m7) {
    const char *dest_addr0 = (*env)->GetDirectBufferAddress(env, m0);
    const char *dest_addr1 = (*env)->GetDirectBufferAddress(env, m1);
    const char *dest_addr2 = (*env)->GetDirectBufferAddress(env, m2);
    const char *dest_addr3 = (*env)->GetDirectBufferAddress(env, m3);
    const char *dest_addr4 = (*env)->GetDirectBufferAddress(env, m4);
    const char *dest_addr5 = (*env)->GetDirectBufferAddress(env, m5);
    const char *dest_addr6 = (*env)->GetDirectBufferAddress(env, m6);
    const char *dest_addr7 = (*env)->GetDirectBufferAddress(env, m7);

    for (i = 0; i < LEN; i++) {
        dest0[i] = dest_addr0[i];
        dest1[i] = dest_addr1[i];
        dest2[i] = dest_addr2[i];
        dest3[i] = dest_addr3[i];
        dest4[i] = dest_addr4[i];
        dest5[i] = dest_addr5[i];
        dest6[i] = dest_addr6[i];
        dest7[i] = dest_addr7[i];
    }

    float r[8] = {0.0};
    dot256(r, src, dest0, dest1, dest2, dest3, dest4, dest5, dest6, dest7);

    result[0] = (jint) r[0];
    result[1] = (jint) r[1];
    result[2] = (jint) r[2];
    result[3] = (jint) r[3];
    result[4] = (jint) r[4];
    result[5] = (jint) r[5];
    result[6] = (jint) r[6];
    result[7] = (jint) r[7];
    /*printf("c result byte: ");
    for (int i = 0; i < 8; i++)
        printf("%d, ", result[i]);
    printf("\n");*/
}

