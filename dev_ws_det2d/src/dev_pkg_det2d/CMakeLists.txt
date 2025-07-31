cmake_minimum_required(VERSION 3.5)
project(dev_pkg_det2d)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(
  -Wall -Wextra -Wpedantic
  -Wno-pedantic -Wno-sign-compare -Wno-unused-function
)

set(LIB_ARCH "aarch64")
set(DEVICE_NAME "RK3588")

set(RKNN_API_PATH ${CMAKE_CURRENT_SOURCE_DIR}/librknn_api)
set(RKNN_API_INCLUDE ${RKNN_API_PATH}/include)
set(RKNN_API_LIB ${RKNN_API_PATH}/${LIB_ARCH}/librknnrt.so)

set(RGA_PATH ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/rga/${DEVICE_NAME})
set(RGA_INCLUDE ${RGA_PATH}/include)
set(RGA_LIB ${RGA_PATH}/lib/Linux/aarch64/librga.a)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
  ${OpenCV_INCLUDE_DIRS}
  ${RKNN_API_INCLUDE}
  ${RGA_INCLUDE}
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)

add_library(rknn_engine SHARED src/engine/rknn_engine.cpp)
target_link_libraries(rknn_engine ${RKNN_API_LIB})
install(TARGETS rknn_engine DESTINATION lib)

add_library(nn_process_yolov8 SHARED
  src/process/preprocess.cpp
  src/process/yolov8_postprocess.cpp
)
target_link_libraries(nn_process_yolov8 ${OpenCV_LIBS} ${RGA_LIB})
install(TARGETS nn_process_yolov8 DESTINATION lib)

add_library(yolov8_lib SHARED src/task/yolov8.cpp)
target_link_libraries(yolov8_lib rknn_engine nn_process_yolov8)
install(TARGETS yolov8_lib DESTINATION lib)

add_library(draw_lib SHARED src/draw/cv_draw.cpp)
target_link_libraries(draw_lib ${OpenCV_LIBS})
install(TARGETS draw_lib DESTINATION lib)

add_executable(yolov8_img_test src/yolov8_img_test.cpp)
target_link_libraries(yolov8_img_test draw_lib yolov8_lib)
install(TARGETS yolov8_img_test DESTINATION lib/${PROJECT_NAME})

add_executable(yolov8_thread_pool_test src/yolov8_thread_pool_test.cpp)
target_link_libraries(yolov8_thread_pool_test draw_lib yolov8_lib pthread)
install(TARGETS yolov8_thread_pool_test DESTINATION lib/${PROJECT_NAME})

ament_package()