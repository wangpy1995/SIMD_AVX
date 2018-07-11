//
// Created by wpy on 18-7-12.
//

#include "include/jni_StaticInit.h"
#include <stdlib.h>

struct _X {
    char *str;
};

static struct _X x[1];

static volatile long ref_count = 0;

/*
 * Class:     jni_StaticInit
 * Method:    initHandle
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_jni_StaticInit_initHandle
        () {
    ++ref_count;
    if (x[0].str == NULL) {
        printf("init x->str\n");
        x[0].str = malloc(sizeof(char) * 10);
        x[0].str[0] = 'x';
        x[0].str[1] = 'y';
        x[0].str[2] = 'z';
    } else {
        printf("exists...\n");
    }
    return (long) x;
}

/*
 * Class:     jni_StaticInit
 * Method:    doSomething
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_jni_StaticInit_doSomething
        () {
    return x[0].str[1];
}

/*
 * Class:     jni_StaticInit
 * Method:    releaseHandle
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_jni_StaticInit_releaseHandle
        () {
    --ref_count;
    if (ref_count == 0) {
        printf("free x->str\n");
        free(x[0].str);
    } else {
        printf("still in use: %ld\n", ref_count);
    }
}