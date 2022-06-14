// Copyright (c) OpenMMLab. All rights reserved.

#include "logger.h"

#include <cstdlib>

#if SPDLOG_VER_MAJOR >= 1
#if defined(__ANDROID__)
#include <spdlog/sinks/android_sink.h>
#else
#include <spdlog/sinks/stdout_color_sinks.h>
#if defined(_MSC_VER)
#include <spdlog/sinks/stdout_sinks.h>
#endif
#endif
#endif

#if SPDLOG_VER_MAJOR >= 1 && SPDLOG_VER_MINOR >= 6
#define MMDEPLOY_SPDLOG_HAS_LOAD_ENV_LEVELS 1
#include <spdlog/cfg/env.h>
#endif

namespace mmdeploy {

static void LoadEnvLevels() {
  auto p = std::getenv("SPDLOG_LEVEL");
  if (p) {
    const std::string str(p);
    if (str == "trace") {
      spdlog::set_level(spdlog::level::trace);
    } else if (str == "debug") {
      spdlog::set_level(spdlog::level::debug);
    } else if (str == "info") {
      spdlog::set_level(spdlog::level::info);
    } else if (str == "warn") {
      spdlog::set_level(spdlog::level::warn);
    } else if (str == "err") {
      spdlog::set_level(spdlog::level::err);
    } else if (str == "critical") {
      spdlog::set_level(spdlog::level::critical);
    } else if (str == "off") {
      spdlog::set_level(spdlog::level::off);
    }
  }
}

std::shared_ptr<spdlog::logger> CreateDefaultLogger() {
#if MMDEPLOY_SPDLOG_HAS_LOAD_ENV_LEVELS
  spdlog::cfg::load_env_levels();
#else
  LoadEnvLevels();
#endif
  constexpr const auto logger_name = "mmdeploy";
#if defined(__ANDROID__)
  return spdlog::android_logger_mt(logger_name);
#elif defined(_MSC_VER)
  return spdlog::stdout_logger_mt(logger_name);
#else
  return spdlog::stdout_color_mt(logger_name);
#endif
}

std::shared_ptr<spdlog::logger> &gLogger() {
  // ! leaky singleton
  static auto ptr = new std::shared_ptr<spdlog::logger>{CreateDefaultLogger()};
  return *ptr;
}

spdlog::logger *GetLogger() { return gLogger().get(); }

void SetLogger(spdlog::logger *logger) {
  gLogger() = std::shared_ptr<spdlog::logger>(logger, [](auto) {});
}

}  // namespace mmdeploy
