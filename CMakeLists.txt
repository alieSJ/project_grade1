cmake_minimum_required(VERSION 3.10)
project(gesture_recognition_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. 找 OpenCV
find_package(OpenCV REQUIRED)

# 2. 找 ZeroMQ
find_package(PkgConfig REQUIRED)
pkg_check_modules(PC_ZMQ REQUIRED libzmq)

# 3. 找 nlohmann_json
find_package(nlohmann_json 3.2.0 REQUIRED)

# 4. 定义可执行文件 —— 一定要在 target_link_libraries 之前！！
add_executable(gesture_recognition
    main.cpp
)

# 5. 包含路径
target_include_directories(gesture_recognition PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${PC_ZMQ_INCLUDE_DIRS}
)

# 6. 链接库
target_link_libraries(gesture_recognition
    ${OpenCV_LIBS}
    ${PC_ZMQ_LIBRARIES}
    nlohmann_json::nlohmann_json
)

