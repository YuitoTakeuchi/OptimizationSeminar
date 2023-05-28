cmake_minimum_required(VERSION 3.25)

set(PART_NAME line_search)

add_executable(${PART_NAME} line_search.cpp)

# includeパスに追加
target_include_directories(${PART_NAME} PRIVATE
    ${PROJECT_SOURCE_DIR}/include
# external library
    ${EIGEN3_INCLUDE_DIR}
)

# ライブラリにリンク
target_link_libraries(${PART_NAME} PRIVATE
# lib
    LineSearch

# external library
    Eigen3::Eigen
)

# コンパイラ関連
target_compile_options(${PART_NAME} PUBLIC -O3 -march=native)

set(PART_NAME gradient_descent)

add_executable(${PART_NAME} gradient_descent.cpp)

# includeパスに追加
target_include_directories(${PART_NAME} PRIVATE
    ${PROJECT_SOURCE_DIR}/include
# external library
    ${EIGEN3_INCLUDE_DIR}
)

# ライブラリにリンク
target_link_libraries(${PART_NAME} PRIVATE
# lib
    LineSearch

# external library
    Eigen3::Eigen
)

# コンパイラ関連
target_compile_options(${PART_NAME} PUBLIC -O3 -march=native)

set(PART_NAME NewtonsMethod)

add_executable(${PART_NAME} NewtonMethod.cpp)

# includeパスに追加
target_include_directories(${PART_NAME} PRIVATE
    ${PROJECT_SOURCE_DIR}/include
# external library
    ${EIGEN3_INCLUDE_DIR}
)

# ライブラリにリンク
target_link_libraries(${PART_NAME} PRIVATE
# lib
    LineSearch

# external library
    Eigen3::Eigen
)

# コンパイラ関連
target_compile_options(${PART_NAME} PUBLIC -O3 -march=native)

set(PART_NAME BFGS)

add_executable(${PART_NAME} BFGS.cpp)

# includeパスに追加
target_include_directories(${PART_NAME} PRIVATE
    ${PROJECT_SOURCE_DIR}/include
# external library
    ${EIGEN3_INCLUDE_DIR}
)

# ライブラリにリンク
target_link_libraries(${PART_NAME} PRIVATE
# lib
    LineSearch

# external library
    Eigen3::Eigen
)

# コンパイラ関連
target_compile_options(${PART_NAME} PUBLIC -O3 -march=native)