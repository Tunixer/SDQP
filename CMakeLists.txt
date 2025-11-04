cmake_minimum_required(VERSION 2.8.3)

project(sdqp)

set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -fPIC")

find_package(Eigen3 REQUIRED)

# Option to enable testing
option(BUILD_SDQP_TEST "Build tests" OFF)
if(BUILD_SDQP_TEST)
    enable_testing()
    find_package(GTest REQUIRED)
endif()

include_directories(
    include
    ${EIGEN3_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME}_example example/sdqp_example.cpp)

# Add test executable if testing is enabled
if(BUILD_SDQP_TEST)
    add_executable(${PROJECT_NAME}_test test/sdqp_test.cpp)
    target_link_libraries(${PROJECT_NAME}_test GTest::gtest GTest::gtest_main)
    add_test(NAME AllTests COMMAND ${PROJECT_NAME}_test)
endif()
