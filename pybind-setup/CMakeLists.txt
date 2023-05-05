cmake_minimum_required(VERSION 3.24)
project(TurnpikeMM)

set(CMAKE_CXX_STANDARD 20)

set(Boost_USE_MULTITHREADED ON)
include_directories(${Boost_INCLUDE_DIRS})

find_package(Python3.10 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(Boost REQUIRED)
find_package(OpenMP)

pybind11_add_module(TurnpikeMM MODULE library.cpp)
target_compile_options(TurnpikeMM PRIVATE -Ofast -march=native -mtune=native -funroll-loops -fopenmp-simd)

target_link_libraries(TurnpikeMM PRIVATE OpenMP::OpenMP_CXX)
target_include_directories(TurnpikeMM PRIVATE "${BOOST_ROOT}/include")
target_link_libraries(TurnpikeMM PRIVATE Eigen3::Eigen ${Boost_LIBRARIES})

if(APPLE)
    set_target_properties(TurnpikeMM PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
endif()

