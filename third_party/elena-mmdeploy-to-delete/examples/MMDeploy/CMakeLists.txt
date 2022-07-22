# Standalone examples.
file(GLOB exe_src "OpFuse.cpp")

get_filename_component(exe_name ${exe_src} NAME_WE)
add_executable(${exe_name} ${exe_src})
target_link_libraries(${exe_name} ${ELENA_LIB})
