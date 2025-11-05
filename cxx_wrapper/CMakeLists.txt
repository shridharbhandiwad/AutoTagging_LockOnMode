cmake_minimum_required(VERSION 3.15)
project(cxxlib)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and pybind11
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Source files
set(SOURCE_FILES
    src/kalman_filter.cpp
    src/gating.cpp
    src/features.cpp
    src/bindings.cpp
)

# Create Python module
pybind11_add_module(cxxlib ${SOURCE_FILES})

# Set optimization flags
if(MSVC)
    target_compile_options(cxxlib PRIVATE /O2)
else()
    target_compile_options(cxxlib PRIVATE -O3)
endif()

# Install target
install(TARGETS cxxlib DESTINATION .)
