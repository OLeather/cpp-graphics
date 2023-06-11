cmake_minimum_required(VERSION 2.8.4)
project(cpp-graphics)

## Project specific configurations go here

# enable c++11 support
set (CMAKE_CXX_FLAGS "-std=c++11 -Wall ${CMAKE_CXX_FLAGS}")


include_directories(/usr/include/X11)

# create the main executable
## add additional .cpp files if needed
add_executable(cpp-graphics main.cpp)

target_link_libraries(cpp-graphics X11)