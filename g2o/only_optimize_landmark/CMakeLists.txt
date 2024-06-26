cmake_minimum_required(VERSION 2.8)
project(vo1)

set(CMAKE_BUILD_TYPE "Release")
add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS "-std=c++11 -O2 ${SSE_FLAGS} -msse4")

find_package(OpenCV 3 REQUIRED)
find_package(fmt REQUIRED)
find_package(g2o REQUIRED)
find_package(Sophus REQUIRED)

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${G2O_INCLUDE_DIRS}
        ${Sophus_INCLUDE_DIRS}
        "/usr/include/eigen3/"
)

add_executable(sim_only_optimization_landmark sim_only_optimization_landmark.cpp)
target_link_libraries(sim_only_optimization_landmark
        g2o_core g2o_stuff
        ${OpenCV_LIBS}
        fmt::fmt)        
