cmake_minimum_required(VERSION 2.8)
project(vo1)

set(CMAKE_BUILD_TYPE "Release")
add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS "-std=c++14 -O2 ${SSE_FLAGS} -msse4")

# Set cmake
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

find_package(Eigen3  REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})


add_executable(sim_visualize sim_visualize.cpp)
target_link_libraries(sim_visualize
	${Pangolin_LIBRARIES})        
