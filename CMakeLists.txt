cmake_minimum_required(VERSION 3.0.0)
project(slam VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 14)

include_directories(include)

set(OpenCV_DIR "/home/yefangok/src/vcpkg/installed/x64-linux/share/opencv")
set(protobuf_DIR "/home/yefangok/src/vcpkg/installed/x64-linux/share/protobuf")
set(quirc_DIR "/home/yefangok/src/vcpkg/installed/x64-linux/share/quirc")
set(pybind11_DIR "/home/yefangok/src/vcpkg/installed/x64-linux/share/pybind11")
set(Python_ROOT_DIR "/home/yefangok/anaconda3")

add_executable(slam ORBextractor.cc main.cpp)
#pybind11_add_module(slam bind1.cpp)

find_package (Python COMPONENTS Interpreter)
find_package(pybind11)
#find_package(fmt CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED)

target_link_libraries(slam PRIVATE opencv_core opencv_imgproc opencv_features2d  opencv_highgui)

