set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
cmake_minimum_required(VERSION 3.0.2)
project(surenaV_gazebo)

find_package(catkin REQUIRED COMPONENTS
  controller_manager
  roscpp
  rospy
  std_msgs
)

find_package(gazebo REQUIRED)

catkin_package()

include_directories(${catkin_INCLUDE_DIRS}
		    ${GAZEBO_INCLUDE_DIRS})

add_executable(gazebo_bridge_node src/gazebo_bridge.cpp)

target_link_libraries(gazebo_bridge_node ${catkin_LIBRARIES}
		      HumanControllerPlugin ${GAZEBO_LIBRARIES})

add_library(HumanControllerPlugin SHARED
  src/HumanControllerPlugin.cpp)
  
link_directories(${GAZEBO_LIBRARY_DIRS})
add_definitions(${GAZEBO_CXX_FLAGS})


