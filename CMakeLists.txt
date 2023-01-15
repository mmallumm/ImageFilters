cmake_minimum_required(VERSION 3.0.0)
project(lab2 CXX)

find_package(OpenCV REQUIRED)

set(HEADERS filters/filters.h)
set(SOURCES filters/filters.cpp)

set(TARGET_NAME filters)

add_library(${TARGET_NAME} ${SOURCES} ${HEADERS})
target_link_libraries(${TARGET_NAME} PUBLIC ${OpenCV_LIBS})

set(HEADERS tickmetr/TickMeter.h)
set(SOURCES tickmetr/TickMeter.cpp)

set(TARGET_NAME tickmetr)

add_library(${TARGET_NAME} ${SOURCES} ${HEADERS})
target_link_libraries(${TARGET_NAME} PUBLIC ${OpenCV_LIBS})

set(TARGET_NAME lab2_main)

add_executable(${TARGET_NAME} main.cpp)
target_link_libraries(${TARGET_NAME} PUBLIC filters tickmetr)
