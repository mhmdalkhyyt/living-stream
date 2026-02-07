cmake_minimum_required(VERSION 3.10)
project(LivingStream VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(living-stream main.cpp)
target_include_directories(living-stream PUBLIC include)

# TODO: Add dependencies as needed
# target_link_libraries(living-stream pthread)
