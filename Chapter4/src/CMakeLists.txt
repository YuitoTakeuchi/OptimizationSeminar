cmake_minimum_required(VERSION 3.25)

set(LIBRARY_NAME LineSearch)

add_library(${LIBRARY_NAME} STATIC LineSearch.cpp)

# includeパスに追加
target_include_directories(${LIBRARY_NAME} PRIVATE
    ${PROJECT_SOURCE_DIR}/include
# external library
    ${EIGEN3_INCLUDE_DIR}
)

# ライブラリにリンク
target_link_libraries(${LIBRARY_NAME} PRIVATE
# lib

# external library
    Eigen3::Eigen
)

# コンパイラ関連
target_compile_options(${LIBRARY_NAME} PUBLIC -O3 -march=native)