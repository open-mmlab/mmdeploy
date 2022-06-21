#ifndef MMDEPLOY_SRC_CORE_TRACER_H_
#define MMDEPLOY_SRC_CORE_TRACER_H_

#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "mmdeploy/core/macro.h"
#include "mmdeploy/core/tensor.h"
#include "mmdeploy/core/types.h"

namespace mmdeploy {
namespace trace {

struct CvtColorParam {
  DataType dtype;
  PixelFormat srt;
  PixelFormat dst;
};

struct CastParam {
  DataType srt;
  DataType dst;
};

struct ResizeParam {
  DataType dtype;
  std::vector<int> size;
  std::string mode;
};

struct CropParam {
  DataType dtype;
  std::vector<int> tlbr;
  std::vector<int> size;
};

struct NormParam {
  DataType dtype;
  std::vector<float> mean;
  std::vector<float> std;
};

struct PadParam {
  DataType dtype;
  float pad_val;
  std::vector<int> tlbr;
  std::vector<int> size;
};

struct HWC2CHWParam {
  DataType dtype;
};

using TransParamType = std::variant<CvtColorParam, CastParam, ResizeParam, PadParam, NormParam,
                                    CropParam, HWC2CHWParam>;

}  // namespace trace

class MMDEPLOY_API Tracer {
 public:
  void TraceResize(const std::string &mode, const std::vector<int> &size, DataType dtype);

  void TraceLoad(const std::string &color_type, bool to_float32, TensorShape shape,
                 PixelFormat pfmt, DataType dtype);

  void TracePad(float pad_val, const std::vector<int> &tlbr, const std::vector<int> &size,
                DataType dtype);

  void TraceNorm(const std::vector<float> &mean, const std::vector<float> &std, bool to_rgb,
                 DataType dtype);

  void TraceCrop(const std::vector<int> &tlbr, const std::vector<int> &size, DataType dtype);

  void TraceDFB(bool to_float, DataType dtype);

  void TraceIm2Tensor(DataType dtype);

 public:
  struct state_t {
    DataType dtype;
    PixelFormat pfmt;
    TensorShape shape;
  };
  using StateType = struct state_t;
  StateType state_;
  std::optional<DataType> common_dtype_;
  std::vector<trace::TransParamType> trans_;
};

}  // namespace mmdeploy

#endif  // MMDEPLOY_SRC_CORE_TRACER_H_
