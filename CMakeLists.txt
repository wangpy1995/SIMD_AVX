cmake_minimum_required(VERSION 3.9)
project(jni_Test C)

set(CMAKE_C_STANDARD 11)

SET(CMAKE_C_FLAGS_DEBUG "$ENV{CMAKE_C_FLAGS_DEBUG} -mavx2 -O0 -Wall -g -ggdb")
SET(CMAKE_C_FLAGS_RELEASE "$ENV{CMAKE_C_FLAGS_RELEASE} -mavx2 -O3 -Wall")


#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx2 -O3")

include_directories(${PROJECT_SOURCE_DIR}/include)

add_library(jni_Test SHARED jni_Test.c)
#add_library(jni_Test SHARED jni_test_ArrayCopyTest.c jni_test_ArrayCopyTest.h int_dot.c)
#add_library(compartor SHARED com_hikvision_algorithm_JNIComparator.h com_hikvision_algorithm_JNIComparator.c)
add_executable(main int_dot.c)
