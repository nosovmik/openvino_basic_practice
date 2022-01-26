cmake_minimum_required (VERSION 3.10)

project(ov_practice)

find_package(OpenCV  REQUIRED)                  # to read input image
find_package(InferenceEngine 2021.4.2 REQUIRED) # to perform image classification using OpenVINO

add_executable(ov_practice ov_practice1.cpp)
target_link_libraries(ov_practice ${OpenCV_LIBRARIES} ${InferenceEngine_LIBRARIES})