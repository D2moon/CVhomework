cmake_minimum_required(VERSION 3.12)
project(Homework1)

# Add OpenCV
set(OpenCV_DIR "D:/ProgramFiles/opencv/build")
find_package(OpenCV CONFIG REQUIRED)

# Add source files
set(SOURCE_FILES
    core/mainBaseline.cpp
)

# Create an executable target
add_executable(Homework1 ${SOURCE_FILES})

# Add link libraries
target_link_libraries(Homework1 PRIVATE
    ${OpenCV_LIBS}
)

