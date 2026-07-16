cmake_minimum_required(VERSION 3.15)
project(MNIST_NeuralNet C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose build type" FORCE)
endif()

if(MSVC)
    add_compile_options(/O2 /Oi /Ot /fp:fast /arch:AVX2)
    
    add_compile_options(/GL /Ob3)
    
    add_link_options(/LTCG)
else()
    add_compile_options(-O3 -march=native)
endif()

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()

include_directories(include)

file(GLOB SOURCES "src/*.c")

add_executable(ffnn-sgd ${SOURCES})

if(UNIX AND NOT APPLE)
    target_link_libraries(ffnn-sgd m)
endif()