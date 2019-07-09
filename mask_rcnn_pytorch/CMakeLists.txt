cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(mask-rcnn)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

set(CMAKE_VERBOSE_MAKEFILE OFF)
set(CMAKE_CXX_STANDARD 14)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../third_party/json/include)

add_compile_options( -fopenmp
                     -pthread
                     -msse3
                     -Wall
                     -Wextra
                     -Wno-unused-parameter)


set(CUDA_NVCC_FLAGS "-std=c++14;")
list(APPEND CUDA_NVCC_FLAGS "--compiler-options;-Wall;")
list(APPEND CUDA_NVCC_FLAGS "--compiler-options;-Wextra;")
list(APPEND CUDA_NVCC_FLAGS "--compiler-options;-Wno-unused-parameter;")
list(APPEND CUDA_NVCC_FLAGS "--compiler-options;-Wno-unknown-pragmas;")
set(CUDA_NVCC_FLAGS_RELEASE "-O3;-DNDEBUG;")
set(CUDA_NVCC_FLAGS_DEBUG "-g;-G;-O0;")
set(CUDA_PROPAGATE_HOST_FLAGS ON)


set(RESOURCE_FILES
                    README.md)

set(SOURCE_FILES
                    debug.h
                    debug.cpp
                    config.h
                    config.cpp
                    maskrcnn.h
                    maskrcnn.cpp
                    visualize.h
                    visualize.cpp
                    imageutils.h
                    imageutils.cpp
                    stateloader.h
                    stateloader.cpp
                    resnet.h
                    resnet.cpp
                    nnutils.h
                    nnutils.cpp
                    fpn.h
                    fpn.cpp
                    anchors.h
                    anchors.cpp
                    rpn.h
                    rpn.cpp
                    classifier.h
                    classifier.cpp
                    roialign.h
                    roialign.cpp
                    mask.h
                    mask.cpp
                    roialign/cuda/crop_and_resize_kernel.cu
                    roialign/cuda/crop_and_resize_kernel.h
                    roialign/crop_and_resize.h
                    roialign/crop_and_resize.cpp
                    roialign/crop_and_resize_gpu.h
                    roialign/crop_and_resize_gpu.cpp
                    nms/cuda/nms_kernel.cu
                    nms/cuda/nms_kernel.h
                    nms/nms.h
                    nms/nms.cpp
                    nms/nms_cuda.h
                    nms/nms_cuda.cpp
                    proposallayer.h
                    proposallayer.cpp
                    detectionlayer.h
                    detectionlayer.cpp
                    detectiontargetlayer.h
                    detectiontargetlayer.cpp
                    nms.h
                    nms.cpp
                    cocoloader.h
                    cocoloader.cpp
                    cocodataset.h
                    cocodataset.cpp
                    boxutils.h
                    boxutils.cpp
                    loss.h
                    loss.cpp
                    statreporter.h
                    statreporter.cpp
                    datasetclasses.h
                    datasetclasses.cpp
                    modelprepare.h
                    modelprepare.cpp
                    inference.h
                    inference.cpp)

set(REQUIRED_LIBS "stdc++fs")
list(APPEND REQUIRED_LIBS ${TORCH_LIBRARIES})
list(APPEND REQUIRED_LIBS ${OpenCV_LIBS})

find_library(GOMP_LIBRARY gomp)

cuda_add_library(mrcnn_lib SHARED ${SOURCE_FILES})
target_link_libraries(mrcnn_lib ${REQUIRED_LIBS} ${GOMP_LIBRARY})
