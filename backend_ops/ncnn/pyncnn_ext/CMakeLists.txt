# ncnn
set(ncnn_DIR ${NCNN_DIR}/build/install/lib/cmake/ncnn)
find_package(ncnn)

# pybind11
set(PYBIND11_DIR ${NCNN_DIR}/python/pybind11)
add_subdirectory(${PYBIND11_DIR} pybind11)

include_directories(${pybind11_INCLUDE_DIR} ${PYTHON_INCLUDE_DIRS})
pybind11_add_module(ncnn_ext ncnn_ext.cpp)
target_link_libraries(ncnn_ext PUBLIC ncnn ${SHARED_TARGET})
set_target_properties(
  ncnn_ext PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                      ${CMAKE_SOURCE_DIR}/mmdeploy/apis/ncnn)
