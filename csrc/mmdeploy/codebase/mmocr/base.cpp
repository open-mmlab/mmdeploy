// Copyright (c) OpenMMLab. All rights reserved.

#include <sstream>

#include "mmdeploy/codebase/mmocr/base.h"

namespace mmdeploy {
namespace mmocr {

using std::string;
using std::vector;

BaseConvertor::BaseConvertor(const Value& cfg) : MMOCR(cfg) {
  auto model = cfg["context"]["model"].get<Model>();
  if (!cfg.contains("params")) {
    MMDEPLOY_ERROR("'params' is required, but it's not in the config");
    throw_exception(eInvalidArgument);
  }
  // BaseConverter
  auto& _cfg = cfg["params"];
  if (_cfg.contains("dict_file")) {
    auto filename = _cfg["dict_file"].get<std::string>();
    auto content = model.ReadFile(filename).value();
    idx2char_ = SplitLines(content);
  } else if (_cfg.contains("dict_list")) {
    from_value(_cfg["dict_list"], idx2char_);
  } else if (_cfg.contains("dict_type")) {
    auto dict_type = _cfg["dict_type"].get<std::string>();
    if (dict_type == "DICT36") {
      idx2char_ = SplitChars(DICT36);
    } else if (dict_type == "DICT90") {
      idx2char_ = SplitChars(DICT90);
    } else {
      MMDEPLOY_ERROR("unknown dict_type: {}", dict_type);
      throw_exception(eInvalidArgument);
    }
  } else {
    MMDEPLOY_ERROR("either dict_file, dict_list or dict_type must be specified");
    throw_exception(eInvalidArgument);
  }

  model_ = model;
}

string BaseConvertor::Idx2Str(const vector<int>& indexes) {
  size_t count = 0;
  for (const auto& idx : indexes) {
    count += idx2char_[idx].size();
  }
  std::string text;
  text.reserve(count);
  for (const auto& idx : indexes) {
    text += idx2char_[idx];
  }
  return text;
}

vector<string> BaseConvertor::SplitLines(const string& s) {
  std::istringstream is(s);
  vector<string> ret;
  string line;
  while (std::getline(is, line)) {
    ret.push_back(std::move(line));
  }
  return ret;
}

vector<string> BaseConvertor::SplitChars(const string& s) {
  vector<string> ret;
  ret.reserve(s.size());
  for (char c : s) {
    ret.push_back({c});
  }
  return ret;
}

}
}