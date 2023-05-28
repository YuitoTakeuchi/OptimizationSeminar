cmake_minimum_required(VERSION 3.25)
include(FetchContent)

set(PROJECT_NAME Optimization)
project (${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 17)

# eigen
find_package(Eigen3 REQUIRED)

# nlohmann json
FetchContent_Declare(
  json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG        v3.10.4
)
FetchContent_MakeAvailable(json)

# library
add_subdirectory(src)

# chapters
add_subdirectory(Chapter3)
add_subdirectory(Chapter4)
