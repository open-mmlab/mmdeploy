// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <sys/time.h>

#include <cstdio>
#include <memory>
#include <string>

class ScopeTimer {
 public:
  ScopeTimer(std::string _name, bool _print = false) : name(_name), print(_print) { begin = now(); }

  ~ScopeTimer() {
    if (!print) {
      return;
    }
    fprintf(stdout, "%s: %ldms\n", name.c_str(), (now() - begin));
  }

  long now() const {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000 + (tv.tv_usec / 1000);
  }

  long cost() const { return now() - begin; }

 private:
  std::string name;
  bool print;
  long begin;
};
