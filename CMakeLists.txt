cmake_minimum_required(VERSION 3.0)
project(perceptron)

find_package( OpenCV REQUIRED )
message(STATUS ${OpenCV_DIR})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp Perceptron.h imageProcessing.h imageProcessing.cpp )
add_executable(perceptron ${SOURCE_FILES})
target_link_libraries( perceptron ${OpenCV_LIBS} )


