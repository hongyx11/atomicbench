cmake_minimum_required(VERSION 3.18.0)

PROJECT(AtomicAddtest CXX CUDA)

set(CMAKE_CXX_STANDARD 14)


# directly make an error if in-source build
if("${PROJECT_SOURCE_DIR}" STREQUAL "${PROJECT_BINARY_DIR}")
    message(FATAL_ERROR "In-source builds are not allowed.\n"
                        "Please create a build directory first and execute cmake configuration from "
                        "this directory. Example: mkdir build && cd build && cmake ..")
endif()

# Set the RPATH config
# --------------------
# use, i.e. don't skip the full RPATH for the build tree
set(CMAKE_SKIP_BUILD_RPATH  FALSE)
# when building, use the install RPATH already
# (automated test will need this)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
# the RPATH to be used when installing
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
find_package(OpenMP REQUIRED)
set (CMAKE_CUDA_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -fopenmp")
find_package(CUDA REQUIRED)
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
set(CMAKE_CUDA_FLAGS "-O3 --shared --compiler-options '-fPIC' --compiler-options -Wall")
message(STATUS "CUDA FLAGS: ${CMAKE_CUDA_FLAGS}")
set(public_link_lib ${public_link_lib} cudart)
add_executable(${PROJECT_NAME} main.cu)
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 60 70 80)
