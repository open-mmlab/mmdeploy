# Copyright (c) OpenMMLab. All rights reserved.
function(set_targets PROJECT_NAME OBJ_TARGET STATIC_TARGET SHARED_TARGET)
    set(${OBJ_TARGET} ${PROJECT_NAME}_obj PARENT_SCOPE)
    set(${STATIC_TARGET} ${PROJECT_NAME}_static PARENT_SCOPE)
    set(${SHARED_TARGET} ${PROJECT_NAME} PARENT_SCOPE)
endfunction()

function(install_targets TARGET_NAMES)
    foreach (TARGET_NAME ${TARGET_NAMES})
        install(TARGETS ${TARGET_NAME}
                ARCHIVE DESTINATION lib
                LIBRARY DESTINATION lib
                RUNTIME DESTINATION bin
                )
    endforeach ()
endfunction()

function(build_target TARGET_NAME TARGET_SRCS)
    add_library(${TARGET_NAME} ${TARGET_SRCS})
    set_target_properties(${TARGET_NAME} PROPERTIES POSITION_INDEPENDENT_CODE 1)
endfunction()

# When the object target ${TARGET_NAME} has more than one source file,
# "${SRCS_VARIABLE}" MUST be passed to ${TARGET_SRCS}. The quotation marks CANNOT be dismissed.
function(build_object_target TARGET_NAME TARGET_SRCS)
    add_library(${TARGET_NAME} OBJECT)
    target_sources(${TARGET_NAME} PRIVATE ${TARGET_SRCS})
    set_target_properties(${TARGET_NAME} PROPERTIES POSITION_INDEPENDENT_CODE 1)
endfunction()

function(build_static_target TARGET_NAME OBJECT_TARGET LINK_TYPE)
    add_library(${TARGET_NAME} STATIC $<TARGET_OBJECTS:${OBJECT_TARGET}>)
    if (${LINK_TYPE} STREQUAL "PRIVATE")
        target_link_libraries(${TARGET_NAME} PRIVATE ${OBJECT_TARGET})
    elseif (${LINK_TYPE} STREQUAL "PUBLIC")
        target_link_libraries(${TARGET_NAME} PUBLIC ${OBJECT_TARGET})
    elseif (${LINK_TYPE} STREQUAL "INTERFACE")
        target_link_libraries(${TARGET_NAME} INTERFACE ${OBJECT_TARGET})
    elseif (${LINK_TYPE} STREQUAL "")
        target_link_libraries(${TARGET_NAME} ${OBJECT_TARGET})
    else ()
        message(FATAL_ERROR "Incorrect link type: ${LINK_TYPE}")
    endif ()
endfunction()

function(build_shared_target TARGET_NAME OBJECT_TARGET LINK_TYPE)
    add_library(${TARGET_NAME} SHARED $<TARGET_OBJECTS:${OBJECT_TARGET}>)
    if (${LINK_TYPE} STREQUAL "PRIVATE")
        target_link_libraries(${TARGET_NAME} PRIVATE ${OBJECT_TARGET})
    elseif (${LINK_TYPE} STREQUAL "PUBLIC")
        target_link_libraries(${TARGET_NAME} PUBLIC ${OBJECT_TARGET})
    elseif (${LINK_TYPE} STREQUAL "INTERFACE")
        target_link_libraries(${TARGET_NAME} INTERFACE ${OBJECT_TARGET})
    elseif (${LINK_TYPE} STREQUAL "")
        target_link_libraries(${TARGET_NAME} ${OBJECT_TARGET})
    else ()
        message(FATAL_ERROR "Incorrect link type: ${LINK_TYPE}")
    endif ()
endfunction()

function(build_module_target TARGET_NAME OBJECT_TARGET LINK_TYPE)
    add_library(${TARGET_NAME} MODULE $<TARGET_OBJECTS:${OBJECT_TARGET}>)
    if (${LINK_TYPE} STREQUAL "PRIVATE")
        target_link_libraries(${TARGET_NAME} PRIVATE ${OBJECT_TARGET})
    elseif (${LINK_TYPE} STREQUAL "PUBLIC")
        target_link_libraries(${TARGET_NAME} PUBLIC ${OBJECT_TARGET})
    elseif (${LINK_TYPE} STREQUAL "INTERFACE")
        target_link_libraries(${TARGET_NAME} INTERFACE ${OBJECT_TARGET})
    elseif (${LINK_TYPE} STREQUAL "")
        target_link_libraries(${TARGET_NAME} ${OBJECT_TARGET})
    else ()
        message(FATAL_ERROR "Incorrect link type: ${LINK_TYPE}")
    endif ()
endfunction()


function(export_target TARGET_NAME)
    target_link_libraries(MMDeployLibs INTERFACE ${TARGET_NAME})
    install(TARGETS ${TARGET_NAME}
            EXPORT MMDeployTargets
            ARCHIVE DESTINATION lib
            LIBRARY DESTINATION lib
            )
endfunction()

function(export_module TARGET_NAME)
    get_target_property(TARGET_TYPE ${TARGET_NAME} TYPE)
    if (${TARGET_TYPE} STREQUAL "STATIC_LIBRARY")
        target_link_libraries(MMDeployStaticModules INTERFACE ${TARGET_NAME})
    elseif (${TARGET_TYPE} STREQUAL "SHARED_LIBRARY")
        target_link_libraries(MMDeployDynamicModules INTERFACE ${TARGET_NAME})
    endif ()
    install(TARGETS ${TARGET_NAME}
            EXPORT MMDeployTargets
            ARCHIVE DESTINATION lib
            LIBRARY DESTINATION lib
            )
endfunction()

function(get_target_list INPUT_TARGETS OUTPUT_TARGETS)
    set(FILTERED_TARGETS)
    foreach (INPUT_TARGET IN LISTS INPUT_TARGETS)
        if (TARGET ${INPUT_TARGET})
            list(APPEND FILTERED_TARGETS ${INPUT_TARGET})
        endif()
    endforeach ()
    set(${OUTPUT_TARGETS} "${FILTERED_TARGETS}" PARENT_SCOPE)
endfunction()
