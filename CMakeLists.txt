cmake_minimum_required(VERSION 3.9)
project(jni_Test C)

set(CMAKE_C_STANDARD 11)

#SET(CMAKE_C_FLAGS_DEBUG "$ENV{CMAKE_C_FLAGS_DEBUG} -mavx2 -Og -Wall -g -ggdb")
#SET(CMAKE_C_FLAGS_RELEASE "$ENV{CMAKE_C_FLAGS_RELEASE} -mavx2 -O3 -Wall")
SET(CMAKE_C_FLAGS "$ENV{CMAKE_C_FLAGS_RELEASE} -mavx2 -O3")


#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx2 -O3")

include_directories(${PROJECT_SOURCE_DIR}/include /opt/amdsdk/v1.0 )
link_directories(${PROJECT_SOURCE_DIR}/libs)


#add_library(StaticInit SHARED jni_StaticInit.c)
#add_library(jni_TestThread SHARED jni_TestThread.c)
#add_library(jni_Test SHARED jni_test_ArrayCopyTest.c jni_test_ArrayCopyTest.h int_dot.c)
#add_library(compartor SHARED com_hikvision_algorithm_JNIComparator.h com_hikvision_algorithm_JNIComparator.c)
add_executable(main int_dot.c)

#target_link_libraries(main OpenCL)