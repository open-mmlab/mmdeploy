// Copyright (c) OpenMMLab. All rights reserved.

#include <algorithm>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "mmdeploy/core/device.h"
#include "mmdeploy/core/model.h"
#include "mmdeploy/core/registry.h"
#include "mmdeploy/core/tensor.h"
#include "mmdeploy/core/utils/device_utils.h"
#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/core/value.h"
#include "mmdeploy/experimental/module_adapter.h"
#include "mmocr.h"

namespace mmdeploy::mmocr {

using std::string;
using std::unordered_map;
using std::unordered_set;
using std::vector;

class BaseConvertor : public MMOCR {
 public:
  explicit BaseConvertor(const Value& cfg);

  string Idx2Str(const vector<int>& indexes);

  virtual Result<Value> operator()(const Value& _data, const Value& _prob) = 0;

 protected:
  static vector<string> SplitLines(const string& s);

  static vector<string> SplitChars(const string& s);

  static constexpr const auto DICT36 = R"(0123456789abcdefghijklmnopqrstuvwxyz)";
  static constexpr const auto DICT90 = R"(0123456789abcdefghijklmnopqrstuvwxyz)"
                                       R"(ABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'())"
                                       R"(*+,-./:;<=>?@[\]_`~)";

  static constexpr const auto kHost = Device(0);

  Model model_;

  int padding_idx_{-1};
  int end_idx_{-1};
  int start_idx_{-1};
  int unknown_idx_{-1};

  unordered_set<int> ignore_indexes_;
  unordered_map<string, int> char2idx_;
  vector<string> idx2char_;

};  // class BaseConvertor

}  // namespace mmdeploy::mmocr
