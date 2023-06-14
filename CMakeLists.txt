cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cpp_graphics LANGUAGES CXX CUDA)

# enable c++11 support
set (CMAKE_CXX_FLAGS "-std=c++11 -Wall ${CMAKE_CXX_FLAGS}")


include_directories(/usr/include/X11)

file(GLOB cpp_graphics_lib_src
     "cpp_graphics/src/*.cpp"
     "cpp_graphics/src/*.cu"
)

add_library(cpp_graphics_lib ${cpp_graphics_lib_src})

target_include_directories(cpp_graphics_lib PUBLIC cpp_graphics/include)
target_link_libraries(cpp_graphics_lib X11)

# create the main executable
## add additional .cpp files if needed
add_executable(cpp_graphics main.cpp)
target_link_libraries(cpp_graphics X11 cpp_graphics_lib)