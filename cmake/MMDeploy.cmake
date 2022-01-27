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


function (_mmdeploy_flatten_modules RETVAL)
    set(_RETVAL)
    foreach (ARG IN LISTS ARGN)
        get_target_property(TYPE ${ARG} TYPE)
        message(STATUS "${ARG} ${TYPE}")
        if (TYPE STREQUAL "INTERFACE_LIBRARY")
            get_target_property(LIBS ${ARG} INTERFACE_LINK_LIBRARIES)
            if (LIBS)
                list(FILTER LIBS EXCLUDE REGEX ::@)
                list(APPEND _RETVAL ${LIBS})
            endif ()
        else ()
            list(APPEND _RETVAL ${ARG})
        endif ()
    endforeach ()
    set(${RETVAL} ${_RETVAL} PARENT_SCOPE)
endfunction ()


function (mmdeploy_load_static NAME)
    if (MSVC)
        target_link_libraries(${NAME} PRIVATE ${ARGN})
    else ()
        _mmdeploy_flatten_modules(_MODULE_LIST ${ARGN})
        target_link_libraries(${NAME} PRIVATE
                -Wl,--whole-archive
                ${_MODULE_LIST}
                -Wl,--no-whole-archive)
    endif ()
endfunction ()

function (mmdeploy_load_dynamic NAME)
    _mmdeploy_flatten_modules(_MODULE_LIST ${ARGN})
    if (MSVC)
        if (NOT _MODULE_LIST)
            return ()
        endif ()
        # MSVC has nothing like "-Wl,--no-as-needed ... -Wl,--as-needed", as a
        # workaround we build a static module which loads the dynamic modules
        set(_MODULE_STR ${_MODULE_LIST})
        list(TRANSFORM _MODULE_STR REPLACE "(.+)" "\"\\1\"")
        string(JOIN ",\n        " _MODULE_STR ${_MODULE_STR})
        set(_MMDEPLOY_DYNAMIC_MODULES ${_MODULE_STR})

        set(_LOADER ${NAME}_loader)

        add_dependencies(${NAME} ${_MODULE_LIST})

        configure_file(
                ${CMAKE_SOURCE_DIR}/csrc/loader/loader.cpp.in
                ${CMAKE_BINARY_DIR}/${_LOADER}.cpp)

        mmdeploy_add_module(${_LOADER} STATIC EXCLUDE ${CMAKE_BINARY_DIR}/${_LOADER}.cpp)
        mmdeploy_load_static(${NAME} ${_LOADER})
    else ()
        target_link_libraries(${NAME} PRIVATE
                -Wl,--no-as-needed
                ${_MODULE_LIST}
                -Wl,--as-needed)
    endif ()
endfunction ()
