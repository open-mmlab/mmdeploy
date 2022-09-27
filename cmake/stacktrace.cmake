# Copyright (c) OpenMMLab. All rights reserved.
find_package(Boost 1.65 COMPONENTS stacktrace_backtrace)
if (Boost_FOUND)
    target_link_libraries(mmdeploy_core PUBLIC Boost::stacktrace_backtrace)
    target_compile_definitions(mmdeploy_core PUBLIC -DMMDEPLOY_STATUS_USE_STACKTRACE=1)
endif()
