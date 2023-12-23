cmake_minimum_required(VERSION 3.24)
project(Homework2)

# Add OpenCV
set(OpenCV_DIR "D:/ProgramFiles/opencv/build")
find_package(OpenCV CONFIG REQUIRED)

# Add source files
set(SOURCE_FILES core/main.cpp)

# Create an executable target
add_executable(Homework2 ${SOURCE_FILES})

# Add link libraries
target_link_libraries(Homework2 PRIVATE
    ${OpenCV_LIBS}
)

