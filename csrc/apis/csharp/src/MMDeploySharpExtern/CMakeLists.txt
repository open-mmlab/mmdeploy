cmake_minimum_required (VERSION 3.15)

project ("MMDeploySharpExtern")

find_package(MMDeploy REQUIRED)

file(GLOB SRCS "*.c" "*.cpp")
add_library(${PROJECT_NAME} SHARED ${SRCS})

mmdeploy_load_static(${PROJECT_NAME} MMDeployStaticModules)
mmdeploy_load_dynamic(${PROJECT_NAME} MMDeployDynamicModules)
target_link_libraries(${PROJECT_NAME} PRIVATE MMDeployLibs ${OpenCV_LIBS})