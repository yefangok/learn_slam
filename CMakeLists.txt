cmake_minimum_required(VERSION 3.0.0)
project(slam VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 14)

include_directories(include)
include_directories(/usr/include/suitesparse)
#link_directories(/usr/lib/x86_64-linux-gnu)

set(OpenCV_DIR "/home/yefangok/src/vcpkg/installed/x64-linux/share/opencv")
set(protobuf_DIR "/home/yefangok/src/vcpkg/installed/x64-linux/share/protobuf")
set(quirc_DIR "/home/yefangok/src/vcpkg/installed/x64-linux/share/quirc")
set(pybind11_DIR "/home/yefangok/src/vcpkg/installed/x64-linux/share/pybind11")
#set(g2o_DIR "/home/yefangok/src/g2o/build/generated")
set(Python_ROOT_DIR "/home/yefangok/anaconda3")

add_executable(slam ORBextractor.cc main.cc)
#pybind11_add_module(slam bind1.cpp)

find_package (Python COMPONENTS Interpreter)
find_package(pybind11)
#find_package(fmt CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED)
find_package(g2o REQUIRED)

target_link_libraries(slam PRIVATE 
    opencv_core 
    opencv_imgproc 
    opencv_features2d  
    opencv_highgui 
    g2o_core 
    g2o_stuff 
    g2o_types_slam3d 
    g2o_types_sba  
    g2o_csparse_extension 
    g2o_solver_csparse
    g2o_solver_cholmod
    cholmod
)

