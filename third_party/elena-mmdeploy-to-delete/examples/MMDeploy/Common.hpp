#include <iostream>

#include "api.h"

using namespace ir;

enum Target { CPU = 1, CUDA = 2 };
enum Dtype { Uint8 = 1, Int32 = 2, Float32 = 3 };
enum Format { BGR = 1, RGB = 2, GRAY = 3, BGRA = 4, NV12 = 5, NV21 = 6 };
enum Interpolation { Bilinear = 1, Nearest = 2 };

namespace Common {
std::vector<std::vector<float>> NV2BGR_params{
    {1.164, 2.018, 0}, {1.164, -0.813, -0.391}, {1.164, 0, 1.596}};  // BGR YUV

void GenerateChannelAndInput(Format format, Dtype dtype, ir::ExprPtr h,
                             ir::ExprPtr w, ir::ExprPtr &c,
                             ir::TensorVarPtr &img) {
  switch (format) {
    case BGR:
      c = api::constant<uint64_t>(3);
      break;
    case RGB:
      c = api::constant<uint64_t>(3);
      break;
    case GRAY:
      c = api::constant<uint64_t>(1);
      break;
    case BGRA:
      c = api::constant<uint64_t>(4);
      break;
    case NV12:
      c = api::constant<uint64_t>(1);
      break;
    case NV21:
      c = api::constant<uint64_t>(1);
      break;
    default:
      break;
  }
  switch (dtype) {
    case Uint8:
      img = api::placeholder<uint8_t>({h, w, c}, "input");
      break;
    case Int32:
      img = api::placeholder<int32_t>({h, w, c}, "input");
      break;
    case Float32:
      img = api::placeholder<float>({h, w, c}, "input");
      break;
    default:
      ELENA_ABORT("dtype not support");
      break;
  }
}

ir::TensorVarPtr CastFloat(const std::vector<ir::ExprPtr> &shape,
                           ir::Array<ir::IterVar> iter_vars,
                           ir::TensorVarPtr input,
                           const std::string &name = "CastFloat") {
  ELENA_ASSERT(
      ptr_cast<Const<uint64_t>>(shape[2])->get_value() ==
          ptr_cast<Const<uint64_t>>(input->shape->element[2])->get_value(),
      "CastFloat");

  return api::compute(shape, iter_vars,
                      cast((*input)(iter_vars[0], iter_vars[1], iter_vars[2]),
                           ir::ScalarType::Float32),
                      "CastFloat");
}

ir::TensorVarPtr BGRMean(std::vector<float> &mean,
                         const std::string &name = "BGRMean") {
  ELENA_ASSERT(mean.size() == 3, "BGRMean");

  std::vector<ir::ExprPtr> mean_shape{api::constant<uint64_t>(mean.size())};
  auto iter_mean = api::construct_indices(mean_shape);

  auto C0 = api::logical::eq(iter_mean[0], api::constant<int>(0));
  auto C1 = api::logical::eq(iter_mean[0], api::constant<int>(1));
  return api::compute(
      mean_shape, iter_mean,
      api::if_then_else(C0, api::constant<float>(mean[0]),
                        api::if_then_else(C1, api::constant<float>(mean[1]),
                                          api::constant<float>(mean[2]))),
      "BGR_mean");
}

ir::TensorVarPtr BGRMean(std::vector<ir::ExprPtr> &mean,
                         const std::string &name = "BGRMean") {
  ELENA_ASSERT(mean.size() == 3, "BGRMean");

  std::vector<ir::ExprPtr> mean_shape{api::constant<uint64_t>(mean.size())};
  auto iter_mean = api::construct_indices(mean_shape);

  auto C0 = api::logical::eq(iter_mean[0], api::constant<int>(0));
  auto C1 = api::logical::eq(iter_mean[0], api::constant<int>(1));
  return api::compute(
      mean_shape, iter_mean,
      api::if_then_else(C0, mean[0], api::if_then_else(C1, mean[1], mean[2])),
      "BGR_mean");
}

ir::TensorVarPtr GrayMean(std::vector<float> &mean,
                          const std::string &name = "GrayMean") {
  ELENA_ASSERT(mean.size() == 1, "GrayMean");

  std::vector<ir::ExprPtr> mean_shape{api::constant<uint64_t>(mean.size())};
  auto iter_mean = api::construct_indices(mean_shape);

  return api::compute(mean_shape, iter_mean, api::constant<float>(mean[0]),
                      "Gray_mean");
}

ir::TensorVarPtr GrayMean(std::vector<ir::ExprPtr> &mean,
                          const std::string &name = "GrayMean") {
  ELENA_ASSERT(mean.size() >= 1, "GrayMean");

  std::vector<ir::ExprPtr> mean_shape{api::constant<uint64_t>(mean.size())};
  auto iter_mean = api::construct_indices(mean_shape);

  return api::compute(mean_shape, iter_mean, mean[0], "Gray_mean");
}

ir::TensorVarPtr BGRStd(std::vector<float> &std,
                        const std::string &name = "BGRStd") {
  ELENA_ASSERT(std.size() == 3, "BGRStd");

  std::vector<ir::ExprPtr> std_shape{api::constant<uint64_t>(std.size())};
  auto iter_std = api::construct_indices(std_shape);

  auto C0 = api::logical::eq(iter_std[0], api::constant<int>(0));
  auto C1 = api::logical::eq(iter_std[0], api::constant<int>(1));
  return api::compute(
      std_shape, iter_std,
      api::if_then_else(C0, api::constant<float>(std[0]),
                        api::if_then_else(C1, api::constant<float>(std[1]),
                                          api::constant<float>(std[2]))),
      "BGR_std");
}

ir::TensorVarPtr BGRStd(std::vector<ir::ExprPtr> &std,
                        const std::string &name = "BGRStd") {
  ELENA_ASSERT(std.size() == 3, "BGRStd");

  std::vector<ir::ExprPtr> std_shape{api::constant<uint64_t>(std.size())};
  auto iter_std = api::construct_indices(std_shape);

  auto C0 = api::logical::eq(iter_std[0], api::constant<int>(0));
  auto C1 = api::logical::eq(iter_std[0], api::constant<int>(1));
  return api::compute(
      std_shape, iter_std,
      api::if_then_else(C0, std[0], api::if_then_else(C1, std[1], std[2])),
      "BGR_std");
}

ir::TensorVarPtr GrayStd(std::vector<float> &std,
                         const std::string &name = "GrayStd") {
  ELENA_ASSERT(std.size() == 1, "GrayStd");

  std::vector<ir::ExprPtr> std_shape{api::constant<uint64_t>(std.size())};
  auto iter_std = api::construct_indices(std_shape);

  return api::compute(std_shape, iter_std, api::constant<float>(std[0]),
                      "Gray_std");
}

ir::TensorVarPtr GrayStd(std::vector<ir::ExprPtr> &std,
                         const std::string &name = "GrayStd") {
  ELENA_ASSERT(std.size() >= 1, "GrayStd");

  std::vector<ir::ExprPtr> std_shape{api::constant<uint64_t>(std.size())};
  auto iter_std = api::construct_indices(std_shape);

  return api::compute(std_shape, iter_std, std[0], "Gray_std");
}

ir::TensorVarPtr NV2BGRParams(const std::string &name = "NV2BGRParams") {
  ELENA_ASSERT(NV2BGR_params.size() == 3 && NV2BGR_params[0].size() == 3 &&
                   NV2BGR_params[1].size() == 3 && NV2BGR_params[2].size() == 3,
               "BGRStd");

  std::vector<ir::ExprPtr> NV2BGR_params_shape{
      api::constant<uint64_t>(NV2BGR_params.size()),
      api::constant<uint64_t>(NV2BGR_params[0].size())};
  auto iter_NV2BGR_params = api::construct_indices(NV2BGR_params_shape);

  auto B = api::logical::eq(iter_NV2BGR_params[0], api::constant<int>(0));
  auto G = api::logical::eq(iter_NV2BGR_params[0], api::constant<int>(1));
  auto R = api::logical::eq(iter_NV2BGR_params[0], api::constant<int>(2));

  auto Y = api::logical::eq(iter_NV2BGR_params[1], api::constant<int>(0));
  auto U = api::logical::eq(iter_NV2BGR_params[1], api::constant<int>(1));
  auto V = api::logical::eq(iter_NV2BGR_params[1], api::constant<int>(2));

  std::vector<ir::ExprPtr> condition_BY;
  condition_BY.push_back(B);
  condition_BY.push_back(Y);
  std::vector<ir::ExprPtr> condition_BU;
  condition_BU.push_back(B);
  condition_BU.push_back(U);
  std::vector<ir::ExprPtr> condition_BV;
  condition_BV.push_back(B);
  condition_BV.push_back(V);
  std::vector<ir::ExprPtr> condition_GY;
  condition_GY.push_back(G);
  condition_GY.push_back(Y);
  std::vector<ir::ExprPtr> condition_GU;
  condition_GU.push_back(G);
  condition_GU.push_back(U);
  std::vector<ir::ExprPtr> condition_GV;
  condition_GV.push_back(G);
  condition_GV.push_back(V);
  std::vector<ir::ExprPtr> condition_RY;
  condition_RY.push_back(R);
  condition_RY.push_back(Y);
  std::vector<ir::ExprPtr> condition_RU;
  condition_RU.push_back(R);
  condition_RU.push_back(U);
  std::vector<ir::ExprPtr> condition_RV;
  condition_RV.push_back(R);
  condition_RV.push_back(V);

  return api::compute(
      NV2BGR_params_shape, iter_NV2BGR_params,
      api::if_then_else(
          api::logical::all(condition_BY),
          api::constant<float>(NV2BGR_params[0][0]),
          api::if_then_else(
              api::logical::all(condition_BU),
              api::constant<float>(NV2BGR_params[0][1]),
              api::if_then_else(
                  api::logical::all(condition_BV),
                  api::constant<float>(NV2BGR_params[0][2]),
                  api::if_then_else(
                      api::logical::all(condition_GY),
                      api::constant<float>(NV2BGR_params[1][0]),
                      api::if_then_else(
                          api::logical::all(condition_GU),
                          api::constant<float>(NV2BGR_params[1][1]),
                          api::if_then_else(
                              api::logical::all(condition_GV),
                              api::constant<float>(NV2BGR_params[1][2]),
                              api::if_then_else(
                                  api::logical::all(condition_RY),
                                  api::constant<float>(NV2BGR_params[2][0]),
                                  api::if_then_else(
                                      api::logical::all(condition_RU),
                                      api::constant<float>(NV2BGR_params[2][1]),
                                      api::constant<float>(
                                          NV2BGR_params[2][2]))))))))),
      "NV2BGR_params");
}

std::string delPrelude(std::string code_row) {
  if (code_row[1] != '#')
    return code_row;
  else {
    int code_length = code_row.size();
    int start = 0;
    for (int i = 1; i < code_length; i++)  // find first line not prelude
    {
      if (code_row[i] == '\n')  // the end of the line
      {
        start = i + 1;
        if (code_row[i + 1] != '#')  // not prelude
          break;
      }
    }
    return code_row.substr(start);
  }
}


}  // namespace Common
