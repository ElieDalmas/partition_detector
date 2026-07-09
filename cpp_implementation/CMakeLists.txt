cmake_minimum_required(VERSION 3.16)
project(omr_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)

add_library(omr_core
    src/otsu.cpp
    src/knn.cpp
    src/features.cpp
    src/classifier.cpp
    src/preprocessing.cpp
    src/segmentation.cpp
    src/staff.cpp
    src/pitch.cpp
    src/evaluation.cpp
    src/dataset.cpp
)

target_include_directories(omr_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/third_parties
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(omr_core PUBLIC ${OpenCV_LIBS})

add_executable(omr_pipeline src/main.cpp)
target_link_libraries(omr_pipeline PRIVATE omr_core)

enable_testing()
add_subdirectory(tests)
