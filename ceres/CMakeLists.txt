cmake_minimum_required(VERSION 3.0)  # 建议使用3.0或更高版本
project(vo1)

set(CMAKE_CXX_STANDARD 11)  
set(CMAKE_CXX_STANDARD_REQUIRED ON)  

# 设置构建类型为Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

# 添加SSE标志
add_definitions("-DENABLE_SSE")

# 设置编译器标志
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -msse4")

# 设置可执行文件输出路径
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

# 寻找所需的依赖包
find_package(OpenCV 3 REQUIRED)
find_package(fmt REQUIRED)
find_package(g2o REQUIRED)
find_package(Sophus REQUIRED)
find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)

# 包含所需的头文件路径
include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${G2O_INCLUDE_DIRS}
        ${Sophus_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${CERES_INCLUDE_DIRS}
)

# 添加可执行文件及其链接的库
add_executable(ceresCurveFitting ceresCurveFitting.cpp)
target_link_libraries(ceresCurveFitting
        ${CERES_LIBRARIES}
        ${OpenCV_LIBS}
        fmt::fmt)  

add_executable(pose_estimation_ceres pose_estimation_ceres.cpp)
target_link_libraries(pose_estimation_ceres
        g2o_core g2o_stuff
        ${CERES_LIBRARIES}
        ${OpenCV_LIBS}
        fmt::fmt)  
