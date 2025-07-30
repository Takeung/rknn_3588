cmake_minimum_required(VERSION 3.5)
project(dev_pkg_det2d)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 设置编译器
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# 设置库架构
set(LIB_ARCH "aarch64")
set(DEVICE_NAME "RK3588")
#  rknn_api 文件夹路径
set(RKNN_API_PATH ${CMAKE_CURRENT_SOURCE_DIR}/librknn_api)
#  rknn_api include 路径
set(RKNN_API_INCLUDE_PATH ${RKNN_API_PATH}/include)
#  rknn_api lib 路径
set(RKNN_API_LIB_PATH ${RKNN_API_PATH}/${LIB_ARCH}/librknnrt.so)
# 寻找OpenCV库，使用自定义的OpenCV_DIR
set(3RDPARTY_PATH ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty)
# set(RGA_DIR ${3RDPARTY_PATH}/rga/${DEVICE_NAME})
# set(RGA_LIB ${RGA_DIR}/lib/Linux/${LIB_ARCH}/librga.so)

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

# use custom interfaces
find_package(dev_pkg_interfaces)

# 用来搜索头文件的目录
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${RKNN_API_INCLUDE_PATH}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    # ${RGA_DIR}/include
)


# 构建自定义封装API库
add_library(rknn_engine SHARED src/engine/rknn_engine.cpp) 
# 链接库
target_link_libraries(rknn_engine
    ${RKNN_API_LIB_PATH}
)
install(TARGETS
  rknn_engine
  DESTINATION lib)


# 构建预处理和后处理库 yolov5
add_library(nn_process_yolov5 SHARED
            src/process/preprocess.cpp
            src/process/yolov5_postprocess.cpp
)
# 链接库
target_link_libraries(nn_process_yolov5
    ${OpenCV_LIBS}
    # ${RGA_LIB}
)
install(TARGETS
  nn_process_yolov5
  DESTINATION lib)


# 构建预处理和后处理库 yolov7
add_library(nn_process_yolov7 SHARED
            src/process/preprocess.cpp
            src/process/yolov7_postprocess.cpp
)
# 链接库
target_link_libraries(nn_process_yolov7
    ${OpenCV_LIBS}
    # ${RGA_LIB}
)
install(TARGETS
  nn_process_yolov7
  DESTINATION lib)


# 构建预处理和后处理库 yolov8
add_library(nn_process_yolov8 SHARED
            src/process/preprocess.cpp
            src/process/yolov8_postprocess.cpp
)
# 链接库
target_link_libraries(nn_process_yolov8
    ${OpenCV_LIBS}
    # ${RGA_LIB}
)
install(TARGETS
  nn_process_yolov8
  DESTINATION lib)


# yolov5_lib
add_library(yolov5_lib SHARED src/task/yolov5.cpp)
# 链接库
target_link_libraries(yolov5_lib
    rknn_engine
    nn_process_yolov5
)
install(TARGETS
  yolov5_lib
  DESTINATION lib)

# yolov7_lib
add_library(yolov7_lib SHARED src/task/yolov7.cpp)
# 链接库
target_link_libraries(yolov7_lib
    rknn_engine
    nn_process_yolov7
)
install(TARGETS
  yolov7_lib
  DESTINATION lib)

# yolov8_lib
add_library(yolov8_lib SHARED src/task/yolov8.cpp)
# 链接库
target_link_libraries(yolov8_lib
    rknn_engine
    nn_process_yolov8
)
install(TARGETS
  yolov8_lib
  DESTINATION lib)


# draw_lib
add_library(draw_lib SHARED src/draw/cv_draw.cpp)
# 链接库
target_link_libraries(draw_lib
    ${OpenCV_LIBS}
)
install(TARGETS
  draw_lib
  DESTINATION lib)


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()


# 测试自yolov5 img test
add_executable(yolov5_img_test src/yolov5_img_test.cpp)
# 链接库
target_link_libraries(yolov5_img_test
        draw_lib
        yolov5_lib
)
install(TARGETS
  yolov5_img_test
  DESTINATION lib/${PROJECT_NAME})


# 测试自yolov5 video
add_executable(yolov5_video_test src/yolov5_video_test.cpp)
# 链接库
target_link_libraries(yolov5_video_test
        draw_lib
        yolov5_lib
)
install(TARGETS
  yolov5_video_test
  DESTINATION lib/${PROJECT_NAME})


# 测试自yolov5 thread pool
add_executable(yolov5_thread_pool_test 
    src/yolov5_thread_pool_test.cpp
)
# 链接库
target_link_libraries(yolov5_thread_pool_test
        draw_lib
        yolov5_lib
        -lpthread
)
install(TARGETS
  yolov5_thread_pool_test
  DESTINATION lib/${PROJECT_NAME})


# # tested from yolov5 ros2 interface
# add_executable(yolov5_ros2_interface_test 
#     src/yolov5_ros2_interface_test.cpp
#     src/task/yolo_thread_pool.cpp
# )
# # 链接库
# target_link_libraries(yolov5_ros2_interface_test
#         draw_lib
#         yolov5_lib
#         -lpthread
# )
# ament_target_dependencies(yolov5_ros2_interface_test 
#         rclcpp 
#         sensor_msgs 
#         cv_bridge
#         dev_pkg_interfaces
# )
# install(TARGETS
#   yolov5_ros2_interface_test
#   DESTINATION lib/${PROJECT_NAME})


# 测试自yolov7 img test
add_executable(yolov7_img_test src/yolov7_img_test.cpp)
# 链接库
target_link_libraries(yolov7_img_test
        draw_lib
        yolov7_lib
)
install(TARGETS
  yolov7_img_test
  DESTINATION lib/${PROJECT_NAME})

# 测试自yolov7 thread pool
add_executable(yolov7_thread_pool_test 
    src/yolov7_thread_pool_test.cpp
)
# 链接库
target_link_libraries(yolov7_thread_pool_test
        draw_lib
        yolov7_lib
        -lpthread
)
install(TARGETS
  yolov7_thread_pool_test
  DESTINATION lib/${PROJECT_NAME})


# 测试自yolov8 img test
add_executable(yolov8_img_test src/yolov8_img_test.cpp)
# 链接库
target_link_libraries(yolov8_img_test
        draw_lib
        yolov8_lib
)
install(TARGETS
  yolov8_img_test
  DESTINATION lib/${PROJECT_NAME})

# 测试自yolov8 thread pool
add_executable(yolov8_thread_pool_test 
    src/yolov8_thread_pool_test.cpp
)
# 链接库
target_link_libraries(yolov8_thread_pool_test
        draw_lib
        yolov8_lib
        -lpthread
)
install(TARGETS
  yolov8_thread_pool_test
  DESTINATION lib/${PROJECT_NAME})


ament_package()
