//
// Created by wpy on 18-3-23.
//

#include <jni_Test.h>
#include <immintrin.h>
#include <inline_avx_sum.h>

static int dot(const char *a, const char *b) {
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

JNIEXPORT jint JNICALL Java_jni_JniDot_dotProduct
        (JNIEnv *env, jobject ignore, jobject src, jobject des) {

    const char *a = (char *) (*env)->GetDirectBufferAddress(env, src);
    const char *b = (char *) (*env)->GetDirectBufferAddress(env, des);
    return dot(a, b);
}

JNIEXPORT void JNICALL Java_jni_JniDot_batchDotProduct
        (JNIEnv *env, jobject ignore, jobject src, jobject des, jobject res, jint batch) {

    int *r = (int *) (*env)->GetDirectBufferAddress(env, res);
    const char *a = (char *) (*env)->GetDirectBufferAddress(env, src);
    const char *b = (char *) (*env)->GetDirectBufferAddress(env, des);
    int i;
    for (i = 0; i < batch; ++i) {
        r[i] = dot(a, b);
        a += 512;
    }
    printf("%d\n",r[0]);
}

