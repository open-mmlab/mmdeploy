// Copyright (c) OpenMMLab. All rights reserved.

#include "base_convertor.h"

namespace mmdeploy::mmocr {

using std::string;
using std::unordered_map;
using std::unordered_set;
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

  // Update Dictionary
  bool with_start = _cfg.value("with_start", false);
  bool with_end = _cfg.value("with_end", false);
  bool same_start_end = _cfg.value("same_start_end", false);
  bool with_padding = _cfg.value("with_padding", false);
  bool with_unknown = _cfg.value("with_unknown", false);

  if (with_start && with_end && same_start_end) {
    start_idx_ = static_cast<int>(idx2char_.size());
    end_idx_ = start_idx_;
    string start_end_token = _cfg.value("start_end_token", string("<BOS/EOS>"));
    idx2char_.emplace_back(std::move(start_end_token));
  } else {
    if (with_start) {
      start_idx_ = static_cast<int>(idx2char_.size());
      string start_token = _cfg.value("start_token", string("<BOS>"));
      idx2char_.emplace_back(std::move(start_token));
    }
    if (with_end) {
      end_idx_ = static_cast<int>(idx2char_.size());
      string end_token = _cfg.value("end_token", string("<EOS>"));
      idx2char_.emplace_back(std::move(end_token));
    }
  }
  if (with_padding) {
    padding_idx_ = static_cast<int>(idx2char_.size());
    string padding_token = _cfg.value("padding_token", string("<PAD>"));
    idx2char_.emplace_back(std::move(padding_token));
  }
  if (with_unknown && _cfg.contains("unknown_token") && !_cfg["unknown_token"].is_null()) {
    unknown_idx_ = static_cast<int>(idx2char_.size());
    string unknown_token = _cfg.value("unknown_token", string("<UKN>"));
    idx2char_.emplace_back(unknown_token);
  }

  // char2idx
  for (int i = 0; i < (int)idx2char_.size(); i++) {
    char2idx_[idx2char_[i]] = i;
  }

  vector<string> ignore_chars;
  if (cfg.contains("ignore_chars")) {
    for (int i = 0; i < cfg["ignore_chars"].size(); i++)
      ignore_chars.emplace_back(cfg["ignore_chars"][i].get<string>());
  } else {
    ignore_chars.emplace_back("padding");
  }
  std::map<string, int> mapping_table = {
      {"padding", padding_idx_}, {"end", end_idx_}, {"unknown", unknown_idx_}};
  for (int i = 0; i < ignore_chars.size(); i++) {
    const auto& ignore_char = ignore_chars[i];
    int ignore_idx = -1;

    if (auto it_default = mapping_table.find(ignore_char); it_default != mapping_table.end()) {
      ignore_idx = it_default->second;
    } else if (auto it_candidate = char2idx_.find(ignore_char); it_candidate != char2idx_.end()) {
      ignore_idx = it_candidate->second;
    } else if (with_unknown) {
      ignore_idx = unknown_idx_;
    }

    if (ignore_idx == -1 || (ignore_idx == unknown_idx_ && ignore_char != "unknown")) {
      MMDEPLOY_WARN("{} does not exist in the dictionary", ignore_char);
      continue;
    }
    ignore_indexes_.insert(ignore_idx);
  }
}

string BaseConvertor::Idx2Str(const vector<int>& indexes) {
  size_t count = 0;
  for (const auto& idx : indexes) {
    if (idx >= idx2char_.size()) {
      MMDEPLOY_ERROR("idx exceeds array bounds {} {}", idx, idx2char_.size());
    }
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

}  // namespace mmdeploy::mmocr
