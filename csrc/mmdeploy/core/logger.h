// Copyright (c) OpenMMLab. All rights reserved.

#ifndef CORE_LOG_H
#define CORE_LOG_H

#include <spdlog/spdlog.h>

#include "mmdeploy/core/macro.h"

namespace mmdeploy {

MMDEPLOY_API spdlog::logger *GetLogger();

MMDEPLOY_API void SetLogger(spdlog::logger *logger);

}  // namespace mmdeploy

// Honor spdlog settings if supported
#if defined(SPDLOG_ACTIVE_LEVEL) && defined(SPDLOG_LEVEL_OFF)

#define MMDEPLOY_LEVEL_TRACE SPDLOG_LEVEL_TRACE
#define MMDEPLOY_LEVEL_DEBUG SPDLOG_LEVEL_DEBUG
#define MMDEPLOY_LEVEL_INFO SPDLOG_LEVEL_INFO
#define MMDEPLOY_LEVEL_WARN SPDLOG_LEVEL_WARN
#define MMDEPLOY_LEVEL_ERROR SPDLOG_LEVEL_ERROR
#define MMDEPLOY_LEVEL_CRITICAL SPDLOG_LEVEL_CRITICAL
#define MMDEPLOY_LEVEL_OFF SPDLOG_LEVEL_OFF

#if !defined(MMDEPLOY_ACTIVE_LEVEL)
#define MMDEPLOY_ACTIVE_LEVEL SPDLOG_ACTIVE_LEVEL
#endif

#else

#define MMDEPLOY_LEVEL_TRACE 0
#define MMDEPLOY_LEVEL_DEBUG 1
#define MMDEPLOY_LEVEL_INFO 2
#define MMDEPLOY_LEVEL_WARN 3
#define MMDEPLOY_LEVEL_ERROR 4
#define MMDEPLOY_LEVEL_CRITICAL 5
#define MMDEPLOY_LEVEL_OFF 6

#if !defined(MMDEPLOY_ACTIVE_LEVEL)
#define MMDEPLOY_ACTIVE_LEVEL MMDEPLOY_LEVEL_INFO
#endif

#endif

#ifdef SPDLOG_LOGGER_CALL
#define MMDEPLOY_LOG(level, ...) SPDLOG_LOGGER_CALL(mmdeploy::GetLogger(), level, __VA_ARGS__)
#else
#define MMDEPLOY_LOG(level, ...) mmdeploy::GetLogger()->log(level, __VA_ARGS__)
#endif

#if MMDEPLOY_ACTIVE_LEVEL <= MMDEPLOY_LEVEL_TRACE
#define MMDEPLOY_TRACE(...) MMDEPLOY_LOG(spdlog::level::trace, __VA_ARGS__)
#else
#define MMDEPLOY_TRACE(...) (void)0;
#endif

#if MMDEPLOY_ACTIVE_LEVEL <= MMDEPLOY_LEVEL_DEBUG
#define MMDEPLOY_DEBUG(...) MMDEPLOY_LOG(spdlog::level::debug, __VA_ARGS__)
#else
#define MMDEPLOY_DEBUG(...) (void)0;
#endif

#if MMDEPLOY_ACTIVE_LEVEL <= MMDEPLOY_LEVEL_INFO
#define MMDEPLOY_INFO(...) MMDEPLOY_LOG(spdlog::level::info, __VA_ARGS__)
#else
#define MMDEPLOY_INFO(...) (void)0;
#endif

#if MMDEPLOY_ACTIVE_LEVEL <= MMDEPLOY_LEVEL_WARN
#define MMDEPLOY_WARN(...) MMDEPLOY_LOG(spdlog::level::warn, __VA_ARGS__)
#else
#define MMDEPLOY_WARN(...) (void)0;
#endif

#if MMDEPLOY_ACTIVE_LEVEL <= MMDEPLOY_LEVEL_ERROR
#define MMDEPLOY_ERROR(...) MMDEPLOY_LOG(spdlog::level::err, __VA_ARGS__)
#else
#define MMDEPLOY_ERROR(...) (void)0;
#endif

#if MMDEPLOY_ACTIVE_LEVEL <= MMDEPLOY_LEVEL_CRITICAL
#define MMDEPLOY_CRITICAL(...) MMDEPLOY_LOG(spdlog::level::critical, __VA_ARGS__)
#else
#define MMDEPLOY_CRITICAL(...) (void)0;
#endif

#endif  // !CORE_LOG_H
