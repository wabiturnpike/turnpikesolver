cmake_minimum_required(VERSION 3.24)

project(TurnpikeMM)

set(CMAKE_CXX_STANDARD 20)
find_package(PkgConfig)

pkg_check_modules(BOOST REQUIRED boost)
pkg_check_modules(Eigen3 REQUIRED eigen3)

set(Boost_USE_MULTITHREADED ON)

find_package(OpenMP REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development)


# Find PyBind info
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
find_package(pybind11 REQUIRED)

# Use pybind11_add_module to create the Python module
pybind11_add_module(TurnpikeMM MODULE library.cpp)
target_compile_options(TurnpikeMM PRIVATE -Ofast -march=native -mtune=native -funroll-loops -fopenmp-simd -finput-charset=UTF-8 -fPIC)

# Set the include directories and linker flags
target_include_directories(TurnpikeMM PRIVATE ${BOOST_INCLUDE_DIRS} ${Eigen3_INCLUDE_DIRS} ${Python_INCLUDE_DIRS})
target_link_libraries(TurnpikeMM PRIVATE OpenMP::OpenMP_CXX ${BOOST_LIBRARIES} ${Python_LIBRARIES})
