cmake_minimum_required(VERSION 3.9)
project(jni_Test C)

set(CMAKE_C_STANDARD 11)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx2")

include_directories(${PROJECT_SOURCE_DIR})

add_library(jni_Test SHARED jni_Test.c jni_Test.h)
#add_library(jni_Test SHARED jni_test_ArrayCopyTest.c jni_test_ArrayCopyTest.h int_dot.c)
#add_library(compartor SHARED com_hikvision_algorithm_JNIComparator.h com_hikvision_algorithm_JNIComparator.c)
add_executable(main int_dot.c)
