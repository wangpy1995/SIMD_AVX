#include <memory.h>
#include "include/jni_test_ArrayCopyTest.h"

/*
 * Class:     jni_test_ArrayCopyTest
 * Method:    testGetElementFalse
 * Signature: ([J)V
 */
JNIEXPORT void JNICALL Java_jni_test_ArrayCopyTest_testGetElementFalse
        (JNIEnv *env, jobject obj, jlongArray arr) {
//    printf("start element false\n");
    jlong *a = (*env)->GetLongArrayElements(env, arr, JNI_FALSE);
    jsize size = (*env)->GetArrayLength(env, arr);
    if (NULL == a) {
        printf("null value\n");
    }/* else {
        printf("e: %ld\n", a[size - 1]);
    }*/
//    (*env)->ReleaseLongArrayElements(env,arr,a,JNI_ABORT);
}

/*
 * Class:     jni_test_ArrayCopyTest
 * Method:    testGetElementTrue
 * Signature: ([J)V
 */
JNIEXPORT void JNICALL Java_jni_test_ArrayCopyTest_testGetElementTrue
        (JNIEnv *env, jobject obj, jlongArray arr) {
    printf("start element true\n");
    jboolean *b;
    *b = JNI_TRUE;
    jlong *a = (*env)->GetLongArrayElements(env, arr, b);
    jsize size = (*env)->GetArrayLength(env, arr);
    if (NULL == a) {
        printf("null value\n");
    } else {
        printf("e: %ld\n", a[size - 1]);
    }
//    (*env)->ReleaseLongArrayElements(env,arr,a,JNI_ABORT);
}

/*
 * Class:     jni_test_ArrayCopyTest
 * Method:    testGetArrayRegionOne
 * Signature: ([J)V
 */
JNIEXPORT void JNICALL Java_jni_test_ArrayCopyTest_testGetArrayRegionOne
        (JNIEnv *env, jobject obj, jlongArray arr) {
    jsize size = (*env)->GetArrayLength(env, arr);
    jlong *a = (*env)->GetPrimitiveArrayCritical(env, arr, NULL);
    if (NULL == &a) {
        printf("null value\n");
    } else {
//        printf("%ld\n", a[size - 1]);
    }
    (*env)->ReleasePrimitiveArrayCritical(env, arr, a, 0);
}

/*
 * Class:     jni_test_ArrayCopyTest
 * Method:    testGetArrayRegionAll
 * Signature: ([J)V
 */
JNIEXPORT void JNICALL Java_jni_test_ArrayCopyTest_testGetArrayRegionAll
        (JNIEnv *env, jobject obj, jlongArray arr) {
    printf("start region all\n");
    jsize size = (*env)->GetArrayLength(env, arr);
    jlong a[size];
    memset(a, 0, sizeof(*a) * size);
    (*env)->GetLongArrayRegion(env, arr, 0, size, a);
    if (NULL == &a) {
        printf("null value\n");
    } else {
        printf("%ld\n", a[size - 1]);
    }
//    (*env)->ReleaseLongArrayElements(env,arr,a,JNI_ABORT);
}