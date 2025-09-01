cmake_minimum_required(VERSION 3.10)
project(vleo-aerodynamics-core)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Include directories
include_directories(include)

# Add Eigen library
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
include_directories(${EIGEN3_INCLUDE_DIR})

# Fortran-based NRLMSIS2.1 library
enable_language(Fortran)
add_subdirectory(../nrlmsis2.1_cpp/Fortran ${CMAKE_BINARY_DIR}/nrlmsis_fortran)

# Add tinyobjloader library
add_subdirectory(external/tinyobjloader)
include_directories(external/tinyobjloader)

# Add source files for the library
file(GLOB LIB_SOURCES "src/*.cpp")

# Define the library
add_library(vleo-aerodynamics-core ${LIB_SOURCES})

# Link libraries to the main library
target_link_libraries(vleo-aerodynamics-core Eigen3::Eigen tinyobjloader msis21)

# Add tests
enable_testing()
file(GLOB TEST_SOURCES "tests/*.cpp")
add_executable(tests ${TEST_SOURCES})

# Link the main library and other dependencies to the tests executable
target_link_libraries(tests vleo-aerodynamics-core Eigen3::Eigen tinyobjloader)

add_test(NAME unit_tests COMMAND tests)

# Tools
add_executable(sweep_shuttlecock tools/sweep_shuttlecock.cpp)
target_link_libraries(sweep_shuttlecock vleo-aerodynamics-core Eigen3::Eigen tinyobjloader)

add_executable(assemble_shuttlecock tools/assemble_shuttlecock.cpp)
target_link_libraries(assemble_shuttlecock vleo-aerodynamics-core Eigen3::Eigen tinyobjloader)

add_executable(compare_shuttlecock tools/compare_shuttlecock.cpp)
target_link_libraries(compare_shuttlecock vleo-aerodynamics-core Eigen3::Eigen tinyobjloader)

# C++ version of MATLAB example

# Demo propagator tool
add_executable(propagate_demo tools/propagate_demo.cpp)
target_link_libraries(propagate_demo vleo-aerodynamics-core Eigen3::Eigen tinyobjloader)

# Env demo
add_executable(env_demo tools/env_demo.cpp)
target_link_libraries(env_demo vleo-aerodynamics-core Eigen3::Eigen tinyobjloader)

# C API shared library for Python/ctypes
add_library(shuttlecock_env SHARED src/c_api.cpp)
target_link_libraries(shuttlecock_env vleo-aerodynamics-core Eigen3::Eigen tinyobjloader msis21)
set_target_properties(shuttlecock_env PROPERTIES OUTPUT_NAME "shuttlecock_env")
