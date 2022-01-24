# Copyright (c) OpenMMLab. All rights reserved.

function (mmdeploy_export NAME)
    install(TARGETS ${NAME}
            EXPORT MMDeployTargets
            ARCHIVE DESTINATION lib
            LIBRARY DESTINATION lib
            RUNTIME DESTINATION bin)
endfunction ()


function (mmdeploy_add_library NAME)
    add_library(${NAME} ${ARGN})
    target_compile_definitions(${NAME} PRIVATE -DMMDEPLOY_API_EXPORTS=1)
    get_target_property(_TYPE ${NAME} TYPE)
    if (_TYPE STREQUAL STATIC_LIBRARY)
        set_target_properties(${NAME} PROPERTIES POSITION_INDEPENDENT_CODE 1)
    elseif (_TYPE STREQUAL SHARED_LIBRARY)
    else ()
        message(FATAL_ERROR "unsupported type: ${_TYPE}")
    endif ()
    target_link_libraries(MMDeployLibs INTERFACE ${NAME})
    mmdeploy_export (${NAME})
endfunction ()


function (mmdeploy_add_module NAME)
    cmake_parse_arguments(_MMDEPLOY "EXCLUDE" "" "" ${ARGN})
    add_library(${NAME} ${_MMDEPLOY_UNPARSED_ARGUMENTS})
    get_target_property(_TYPE ${NAME} TYPE)
    target_link_libraries(${NAME} PRIVATE mmdeploy::core)
    if (_TYPE STREQUAL STATIC_LIBRARY)
        set_target_properties(${NAME} PROPERTIES POSITION_INDEPENDENT_CODE 1)
        if (MSVC)
            target_link_options(${NAME} INTERFACE "/WHOLEARCHIVE:${NAME}")
        endif ()
        if (NOT _MMDEPLOY_EXCLUDE)
            target_link_libraries(MMDeployStaticModules INTERFACE ${NAME})
        endif ()
    elseif (_TYPE STREQUAL SHARED_LIBRARY)
        if (NOT _MMDEPLOY_EXCLUDE)
            target_link_libraries(MMDeployDynamicModules INTERFACE ${NAME})
        endif ()
    else ()
        message(FATAL_ERROR "unsupported type: ${_TYPE}")
    endif ()
    if (NOT _MMDEPLOY_EXCLUDE)
        mmdeploy_export(${NAME})
    endif ()
endfunction ()


function (mmdeploy_load_static NAME)
    set(_TARGETS ${ARGN})
    if (NOT MSVC)
        set(_TARGETS "-Wl,--whole-archive ${_TARGETS} -Wl,--no-whole-archive")
    endif ()
    target_link_libraries(${NAME} PRIVATE ${_TARGETS})
endfunction ()

function (mmdeploy_load_dynamic NAME)
    if (MSVC)
        # MSVC has nothing like "-Wl,--no-as-needed ... -Wl,--as-needed", as a
        # workaround we build a static module which loads the dynamic modules
        set(_module_list)
        foreach (module IN LISTS ARGN)
            get_target_property(_TYPE ${module} TYPE)
            if (_TYPE STREQUAL "INTERFACE_LIBRARY")
                get_target_property(_items ${module} INTERFACE_LINK_LIBRARIES)
                list(FILTER _items EXCLUDE REGEX ::@)
                list(APPEND _module_list ${_items})
            else ()
                list(APPEND _module_list ${module})
            endif ()
        endforeach ()

        set(_module_str ${_module_list})
        list(TRANSFORM _module_str REPLACE "(.+)" "\"\\1\"")
        string(JOIN ",\n        " _module_str ${_module_str})
        set(_MMDEPLOY_DYNAMIC_MODULES ${_module_str})
        set(_LOADER ${NAME}_loader)

        if (NOT _module_list)
            return ()
        endif ()

        add_dependencies(${NAME} ${_module_list})

        configure_file(
                ${CMAKE_SOURCE_DIR}/csrc/loader/loader.cpp.in
                ${CMAKE_BINARY_DIR}/${_LOADER}.cpp)

        mmdeploy_add_module(${_LOADER} STATIC EXCLUDE ${CMAKE_BINARY_DIR}/${_LOADER}.cpp)
        mmdeploy_load_static(${NAME} ${_LOADER})
    else ()
        target_link_libraries(${NAME} PRIVATE
                -Wl,--no-as-needed ${ARGN} -Wl,--as-needed)
    endif ()
endfunction ()