//
// Created by wangpengyu6 on 18-3-2.
//

#include "include/com_hikvision_algorithm_JNIComparator.h"
#include <immintrin.h>
#include <avxintrin.h>

//内存对齐操作
#define ALIGNTO(n) __attribute__((aligned(n)))
#define PREFIX_LEN 16
//请求比对对象的地址
char *src;
static const char prefix[] = {'H', 'I', 'K', 'D', 'F', 'R', '3', '2', 'X', '0', '0', '0', '0', '0', '0', '0'};
static const __m128i* head = (const __m128i *) prefix;

JNIEXPORT jboolean JNICALL Java_com_hikvision_algorithm_JNIComparator_updateAddress
        (JNIEnv *env, jobject obj, jobject m1, jobject res) {
    src = (*env)->GetDirectBufferAddress(env, m1);
    __m128i legal_model = _mm_cmpeq_epi8(*head, *(__m128i *) src);
    int64_t *legal = (int64_t *) &legal_model;
    if (legal[0] == 0 || legal[1] == 0) {
        src = NULL;
        return JNI_FALSE;
    } else {
        return JNI_TRUE;
    }
}

/**
 *
 * @param env
 * @param obj
 * @param model
 * @param len
 * @return NULL: illegal model  Int:dot product
 */
JNIEXPORT jint JNICALL Java_com_hikvision_algorithm_JNIComparator_compare
        (JNIEnv *env, jobject obj, jobject model, jint len) {

    if (!src) return 0;
    char *des = (*env)->GetDirectBufferAddress(env, model);

    //check
    __m128i legal_model = _mm_cmpeq_epi8(*head, *(__m128i *) des);
    int64_t *legal = (int64_t *) &legal_model;

    if (legal[0] == 0 || legal[1] == 0) {
        return 0;
    }

    //compare
    int32_t result = 0;
    __m256i tmp_sum = {0};
    int32_t *sum = (int32_t *) &tmp_sum;
    int size = len & 0xfffffff0;

    __m256i s, d;
    for (int i = PREFIX_LEN; i < size; i += 16) {
        s = _mm256_cvtepi8_epi16(*(__m128i *) (src + i));
        d = _mm256_cvtepi8_epi16(*(__m128i *) (des + i));
        tmp_sum = _mm256_add_epi32(tmp_sum, _mm256_madd_epi16(s, d));
    }

    for (int j = size; j < len; ++j) {
        result += src[j] * des[j];
    }

    result += sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] + sum[6] + sum[7];
    return result;
}

