cmake_minimum_required(VERSION 3.10)

# Project name
project(LumberDefectClassifier)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Add the executable
add_executable(defect_classifier
    main.cpp
    ImageTextureFeatures.cpp
    BayesianDefectClassifier.cpp
)

# Link OpenCV libraries
target_link_libraries(defect_classifier ${OpenCV_LIBS})
