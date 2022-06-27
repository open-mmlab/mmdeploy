# Standalone examples.

file(GLOB executable_src "*.cpp")

foreach(exe_src IN LISTS executable_src)
    get_filename_component(exe_name ${exe_src} NAME_WE)
    add_executable(${exe_name} ${exe_src})
    target_link_libraries(${exe_name} ${ELENA_LIB})
endforeach(exe_src)
