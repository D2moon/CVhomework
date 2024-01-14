cmake_minimum_required(VERSION 3.24)
project(Homework3)

# Macro Definition
add_compile_definitions(
    -DTRAIN_MODEL=0     # 0表示训练，1表示测试
    -DTINY_IMAGE=1
    -DBAG_OF_SIFT=0
)

# Add OpenCV
set(OpenCV_DIR "D:/ProgramFiles/opencv/build")
find_package(OpenCV CONFIG REQUIRED)

# Add source files
set(SOURCE_FILES core/main.cpp)

# Create an executable target
add_executable(Homework3 ${SOURCE_FILES})

# Add link libraries
target_link_libraries(Homework3 PRIVATE
    ${OpenCV_LIBS}
)

