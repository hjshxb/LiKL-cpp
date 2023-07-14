cmake_minimum_required(VERSION 3.5)
project(LiKL)

IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release)
ENDIF()

message("CMAKE_BUILD_TYPE " ${CMAKE_BUILD_TYPE})

if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 14)
endif()

message("-- CMAKE_CXX_STANDARD " ${CMAKE_CXX_STANDARD})

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -O3")

# set(Torch_DIR "/home/hxb/Downloads/3rdparty/libtorch-1.11.0+cu113/libtorch/share/cmake/Torch")
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(Eigen3 REQUIRED)
find_package(OpenCV 4 REQUIRED)
find_package(Torch REQUIRED)
find_package(TorchVision REQUIRED)
# find_package(Glog REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(glog REQUIRED libglog)

message("-- OPENCV VERSION ${OpenCV_VERSION}")
message("-- Glog VERSION ${glog_VERSION}")

add_library(${PROJECT_NAME} SHARED "")
target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_link_libraries(${PROJECT_NAME} 
                    ${OpenCV_LIBS} 
                    ${glog_LIBRARIES} 
                    ${TORCH_LIBRARIES} 
                    TorchVision::TorchVision)

add_subdirectory(src/third_party/anms)
add_subdirectory(src/utils)
add_subdirectory(src/feature)
add_subdirectory(src/param)

add_executable(test_likl example/test_likl.cpp)
target_link_libraries(test_likl ${PROJECT_NAME})