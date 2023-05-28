cmake_minimum_required(VERSION 3.25)
include(FetchContent)

set(PROJECT_NAME LineSearchTest)
project (${PROJECT_NAME})

add_executable(${PROJECT_NAME} main.cpp)

# eigen
find_package(Eigen3 REQUIRED)

target_include_directories(${PROJECT_NAME} 
    PUBLIC ${PROJECT_SOURCE_DIR}/include/
    PUBLIC ${EIGEN3_INCLUDE_DIR}
)

add_subdirectory(src)

target_link_libraries(${PROJECT_NAME} PRIVATE
    PRIVATE LineSearch
# external libs
    PUBLIC Eigen3::Eigen
)

# コンパイラ関連
target_compile_options(${PROJECT_NAME} PUBLIC -O3 -march=native)