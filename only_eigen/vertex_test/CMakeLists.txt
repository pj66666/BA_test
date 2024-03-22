cmake_minimum_required(VERSION 2.8)
project(vo1)

set(CMAKE_BUILD_TYPE "Release")
add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS "-std=c++11 -O2 ${SSE_FLAGS} -msse4")

# find_package(OpenCV 3 REQUIRED)
# find_package(fmt REQUIRED)
# find_package(g2o REQUIRED)
# find_package(Sophus REQUIRED)
find_package(Eigen3 REQUIRED)
include_directories(
        # ${OpenCV_INCLUDE_DIRS}
        # ${G2O_INCLUDE_DIRS}
        ${Sophus_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${EIGEN3_INCLUDE_DIR}
)
add_library(v vertex.cpp)
add_executable(main main.cpp)
target_link_libraries(main
        v)        
