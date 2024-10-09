
set(_TARGETS_PATH
    ${CMAKE_INSTALL_PREFIX}/lib/cmake/MMDeploy/MMDeployTargets.cmake)

file(READ ${_TARGETS_PATH} _MMDEPLOY_TARGETS)

string(REGEX REPLACE "::@<0x[a-z0-9]+>" "" _MMDEPLOY_TARGETS_FIXED
                     "${_MMDEPLOY_TARGETS}")

file(WRITE ${_TARGETS_PATH} "${_MMDEPLOY_TARGETS_FIXED}")
