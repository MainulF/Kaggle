cmake_minimum_required(VERSION 4.2)
project(Kaggle)

set(CMAKE_CXX_STANDARD 26)

include_directories(.)

add_executable(Kaggle
        DecisionTree.cpp
        DecisionTree.h
        main.cpp)
