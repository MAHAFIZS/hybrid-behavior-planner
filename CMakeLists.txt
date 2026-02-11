cmake_minimum_required(VERSION 3.16)
project(hybrid_behavior_planner LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(nlohmann_json 3.2.0 REQUIRED)

add_library(sim_env
  sim_env/src/world.cpp
  sim_env/src/vehicle.cpp
  sim_env/src/sensors.cpp
  sim_env/src/scenario.cpp
)
target_include_directories(sim_env PUBLIC sim_env/include)
target_link_libraries(sim_env PUBLIC nlohmann_json::nlohmann_json)

add_library(planner_core
  planner_core/src/fsm.cpp
  planner_core/src/safety.cpp
)
target_include_directories(planner_core PUBLIC planner_core/include)
target_link_libraries(planner_core PUBLIC sim_env)

add_executable(run_fsm apps/run_fsm.cpp)
target_link_libraries(run_fsm PRIVATE sim_env planner_core)

add_executable(rollout_to_dataset apps/rollout_to_dataset.cpp)
target_link_libraries(rollout_to_dataset PRIVATE sim_env planner_core)
