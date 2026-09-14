cmake_minimum_required(VERSION 3.8)
project(vm_vln_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

rosidl_generate_interfaces(
  ${PROJECT_NAME}
    "msg/VlnHybridOutput.msg"
    "srv/GetVlnFrame.srv"
  DEPENDENCIES
    nav_msgs
    sensor_msgs
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()
