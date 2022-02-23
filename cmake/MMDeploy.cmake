# Copyright (c) OpenMMLab. All rights reserved.

function (mmdeploy_export NAME)
    set(_LIB_DIR lib)
    if (MSVC)
        set(_LIB_DIR bin)
    endif ()
    install(TARGETS ${NAME}
            EXPORT MMDeployTargets
            ARCHIVE DESTINATION lib
            LIBRARY DESTINATION ${_LIB_DIR}
            RUNTIME DESTINATION bin)
endfunction ()


function (mmdeploy_add_library NAME)
    cmake_parse_arguments(_MMDEPLOY "EXCLUDE" "" "" ${ARGN})
    add_library(${NAME} ${_MMDEPLOY_UNPARSED_ARGUMENTS})
    target_compile_definitions(${NAME} PRIVATE -DMMDEPLOY_API_EXPORTS=1)
    get_target_property(_TYPE ${NAME} TYPE)
    if (_TYPE STREQUAL STATIC_LIBRARY)
        set_target_properties(${NAME} PROPERTIES POSITION_INDEPENDENT_CODE 1)
    elseif (_TYPE STREQUAL SHARED_LIBRARY)
    else ()
        message(FATAL_ERROR "unsupported type: ${_TYPE}")
    endif ()
    if (NOT _MMDEPLOY_EXCLUDE)
        target_link_libraries(MMDeployLibs INTERFACE ${NAME})
        mmdeploy_export(${NAME})
    endif ()
endfunction ()


function (mmdeploy_add_module NAME)
    # EXCLUDE: exclude from registering & exporting as SDK module
    # LIBRARY: the module is also a library (add_libray with SHARED instead of MODULE)
    cmake_parse_arguments(_MMDEPLOY "EXCLUDE;LIBRARY" "" "" ${ARGN})
    # search for add_library keywords
    cmake_parse_arguments(_KW "STATIC;SHARED;MODULE" "" "" ${_MMDEPLOY_UNPARSED_ARGUMENTS})

    set(_MAYBE_MODULE)
    # no library type specified
    if (NOT (_KW_STATIC OR  _KW_SHARED OR _KW_MODULE))
        # shared but not marked as a library, build module library so that no .lib dependency
        # will be generated for MSVC
        if (MSVC AND BUILD_SHARED_LIBS AND NOT _MMDEPLOY_LIBRARY)
            set(_MAYBE_MODULE MODULE)
        endif ()
    endif ()

    add_library(${NAME} ${_MAYBE_MODULE} ${_MMDEPLOY_UNPARSED_ARGUMENTS})

    # automatically link mmdeploy::core if exists
    if (TARGET mmdeploy::core)
        target_link_libraries(${NAME} PRIVATE mmdeploy::core)
    endif ()

    # export public symbols when marked as a library
    if (_MMDEPLOY_LIBRARY)
        target_compile_definitions(${NAME} PRIVATE -DMMDEPLOY_API_EXPORTS=1)
    endif ()

    get_target_property(_TYPE ${NAME} TYPE)
    if (_TYPE STREQUAL STATIC_LIBRARY)
        set_target_properties(${NAME} PROPERTIES POSITION_INDEPENDENT_CODE 1)
        if (MSVC)
            target_link_options(${NAME} INTERFACE "/WHOLEARCHIVE:${NAME}")
        endif ()
        # register static modules
        if (NOT _MMDEPLOY_EXCLUDE)
            target_link_libraries(MMDeployStaticModules INTERFACE ${NAME})
        endif ()
    elseif (_TYPE STREQUAL SHARED_LIBRARY OR _TYPE STREQUAL MODULE_LIBRARY)
        # register dynamic modules
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
        if (TYPE STREQUAL "INTERFACE_LIBRARY")
            get_target_property(LIBS ${ARG} INTERFACE_LINK_LIBRARIES)
            if (LIBS)
                # pattern for 3.17+
                list(FILTER LIBS EXCLUDE REGEX "^::@")
                # pattern for 3.13-3.16
                list(TRANSFORM LIBS REPLACE "(.+)::@.*" "\\1")
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

        set(_LOADER_NAME ${NAME}_loader)

        add_dependencies(${NAME} ${_MODULE_LIST})

        set(_LOADER_PATH ${CMAKE_BINARY_DIR}/${_LOADER_NAME}.cpp)
        # ! CMAKE_CURRENT_FUNCTION_LIST_DIR requires cmake 3.17+
        configure_file(
                ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/loader.cpp.in
                ${_LOADER_PATH})

        mmdeploy_add_module(${_LOADER_NAME} STATIC EXCLUDE ${_LOADER_PATH})
        mmdeploy_load_static(${NAME} ${_LOADER_NAME})
    else ()
        target_link_libraries(${NAME} PRIVATE
                -Wl,--no-as-needed
                ${_MODULE_LIST}
                -Wl,--as-needed)
    endif ()
endfunction ()
