cmake_minimum_required (VERSION 3.10)

project(ov_practice_step4)

find_package(OpenCV  REQUIRED)                  # to read input image
find_package(InferenceEngine 2021.4.2 REQUIRED) # to perform image classification using OpenVINO

add_executable(ov_practice_step4 ov_practice4.cpp)
target_link_libraries(ov_practice_step4 ${OpenCV_LIBRARIES} ${InferenceEngine_LIBRARIES})