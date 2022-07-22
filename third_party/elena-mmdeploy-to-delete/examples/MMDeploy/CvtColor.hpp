#include <iostream>

#include "api.h"

using namespace ir;

namespace CvtColor {

ir::TensorVarPtr BGR2RGB(const std::vector<ir::ExprPtr> &shape,
                         ir::Array<ir::IterVar> iter_vars,
                         ir::TensorVarPtr input,
                         const std::string &name = "bgr2rgb") {
  ELENA_ASSERT(ptr_cast<Const<uint64_t>>(shape[2])->get_value() == 3,
               "bgr2rgb");
  ELENA_ASSERT(
      ptr_cast<Const<uint64_t>>(input->shape->element[2])->get_value() == 3,
      "bgr2rgb");

  return api::compute(shape, iter_vars,
                      (*input)(iter_vars[0], iter_vars[1],
                               api::constant<int>(2) - iter_vars[2]),
                      name);
}

ir::TensorVarPtr BGR2RGB(const std::vector<ir::ExprPtr> &shape,
                         ir::TensorVarPtr input,
                         const std::string &name = "bgr2rgb") {
  ELENA_ASSERT(ptr_cast<Const<uint64_t>>(shape[2])->get_value() == 3,
               "bgr2rgb");
  ELENA_ASSERT(
      ptr_cast<Const<uint64_t>>(input->shape->element[2])->get_value() == 3,
      "bgr2rgb");

  auto iter_vars = api::construct_indices(shape);
  return api::compute(shape, iter_vars,
                      (*input)(iter_vars[0], iter_vars[1],
                               api::constant<int>(2) - iter_vars[2]),
                      name);
}

ir::TensorVarPtr RGB2BGR(const std::vector<ir::ExprPtr> &shape,
                         ir::TensorVarPtr input,
                         const std::string &name = "rgb2bgr") {
  ELENA_ASSERT(ptr_cast<Const<uint64_t>>(shape[2])->get_value() == 3,
               "rgb2bgr");
  ELENA_ASSERT(
      ptr_cast<Const<uint64_t>>(input->shape->element[2])->get_value() == 3,
      "rgb2bgr");

  auto iter_vars = api::construct_indices(shape);
  return api::compute(shape, iter_vars,
                      (*input)(iter_vars[0], iter_vars[1],
                               api::constant<int>(2) - iter_vars[2]),
                      name);
}

ir::TensorVarPtr GRAY2BGR(const std::vector<ir::ExprPtr> &shape,
                          ir::TensorVarPtr input,
                          const std::string &name = "gray2bgr") {
  ELENA_ASSERT(ptr_cast<Const<uint64_t>>(shape[2])->get_value() == 3,
               "gray2bgr");
  ELENA_ASSERT(
      ptr_cast<Const<uint64_t>>(input->shape->element[2])->get_value() == 1,
      "gray2bgr");

  auto iter_vars = api::construct_indices(shape);
  return api::compute(
      shape, iter_vars,
      (*input)(iter_vars[0], iter_vars[1], api::constant<int>(0)), name);
}

ir::TensorVarPtr BGRA2BGR(const std::vector<ir::ExprPtr> &shape,
                          ir::TensorVarPtr input,
                          const std::string &name = "bgra2bgr") {
  ELENA_ASSERT(ptr_cast<Const<uint64_t>>(shape[2])->get_value() == 3,
               "bgra2bgr");
  ELENA_ASSERT(
      ptr_cast<Const<uint64_t>>(input->shape->element[2])->get_value() == 4,
      "bgra2bgr");

  auto iter_vars = api::construct_indices(shape);
  return api::compute(shape, iter_vars,
                      (*input)(iter_vars[0], iter_vars[1], iter_vars[2]), name);
}

ir::TensorVarPtr BGR2GRAY(const std::vector<ir::ExprPtr> &shape,
                          ir::TensorVarPtr input,
                          const std::string &name = "bgr2gray") {
  ELENA_ASSERT(ptr_cast<Const<uint64_t>>(shape[2])->get_value() == 1,
               "bgr2gray");
  ELENA_ASSERT(
      ptr_cast<Const<uint64_t>>(input->shape->element[2])->get_value() == 3,
      "bgr2gray");

  auto iter_vars = api::construct_indices(shape);
  return api::compute(
      shape, iter_vars,
      cast(api::constant<float>(0.1140) * (*input)(iter_vars[0], iter_vars[1],
                                                   api::constant<uint64_t>(0)) +
               api::constant<float>(0.5870) *
                   (*input)(iter_vars[0], iter_vars[1],
                            api::constant<uint64_t>(1)) +
               api::constant<float>(0.2989) *
                   (*input)(iter_vars[0], iter_vars[1],
                            api::constant<uint64_t>(2)),
           input->get_dtype()),
      name);
}

ir::TensorVarPtr RGB2GRAY(const std::vector<ir::ExprPtr> &shape,
                          ir::TensorVarPtr input,
                          const std::string &name = "rgb2gray") {
  ELENA_ASSERT(ptr_cast<Const<uint64_t>>(shape[2])->get_value() == 1,
               "bgr2gray");
  ELENA_ASSERT(
      ptr_cast<Const<uint64_t>>(input->shape->element[2])->get_value() == 3,
      "bgr2gray");

  auto iter_vars = api::construct_indices(shape);
  return api::compute(
      shape, iter_vars,
      cast(api::constant<float>(0.1140) * (*input)(iter_vars[0], iter_vars[1],
                                                   api::constant<uint64_t>(0)) +
               api::constant<float>(0.5870) *
                   (*input)(iter_vars[0], iter_vars[1],
                            api::constant<uint64_t>(1)) +
               api::constant<float>(0.2989) *
                   (*input)(iter_vars[0], iter_vars[1],
                            api::constant<uint64_t>(2)),
           input->get_dtype()),
      name);
}

ir::TensorVarPtr BGRA2GRAY(const std::vector<ir::ExprPtr> &shape,
                           ir::TensorVarPtr input,
                           const std::string &name = "bgra2gray") {
  ELENA_ASSERT(ptr_cast<Const<uint64_t>>(shape[2])->get_value() == 1,
               "bgra2gray");
  ELENA_ASSERT(
      ptr_cast<Const<uint64_t>>(input->shape->element[2])->get_value() == 4,
      "bgra2gray");

  auto iter_vars = api::construct_indices(shape);
  return api::compute(
      shape, iter_vars,
      cast(api::constant<float>(0.1140) * (*input)(iter_vars[0], iter_vars[1],
                                                   api::constant<uint64_t>(0)) +
               api::constant<float>(0.5870) *
                   (*input)(iter_vars[0], iter_vars[1],
                            api::constant<uint64_t>(1)) +
               api::constant<float>(0.2989) *
                   (*input)(iter_vars[0], iter_vars[1],
                            api::constant<uint64_t>(2)),
           input->get_dtype()),
      name);
}

ir::TensorVarPtr BGR2NV12(const std::vector<ir::ExprPtr> &shape,
                          ir::TensorVarPtr input,
                          const std::string &name = "bgr2nv12") {
  ELENA_ASSERT(ptr_cast<Const<uint64_t>>(shape[2])->get_value() == 1,
               "bgr2nv12");
  ELENA_ASSERT(
      ptr_cast<Const<uint64_t>>(input->shape->element[2])->get_value() == 3,
      "bgr2nv12");

  auto iter_vars = api::construct_indices(shape);

  auto Y_h =
      cast(shape[0] * api::constant<uint64_t>(2) / api::constant<uint64_t>(3),
           input->get_dtype());
  auto Y_w =
      cast(shape[1] * api::constant<uint64_t>(2) / api::constant<uint64_t>(3),
           input->get_dtype());
  std::vector<ir::ExprPtr> condition_Y;
  condition_Y.push_back(api::logical::lt(iter_vars[0], Y_h));
  condition_Y.push_back(api::logical::lt(iter_vars[1], Y_w));

  std::vector<ir::ExprPtr> condition_U;
  condition_U.push_back(api::logical::eq(
      iter_vars[1] % api::constant<uint64_t>(2), api::constant<uint64_t>(0)));

  return api::compute(
      shape, iter_vars,
      api::if_then_else(
          api::logical::all(condition_Y),
          // Y
          cast(api::constant<float>(0.0980) *
                       (*input)(iter_vars[0], iter_vars[1],
                                api::constant<uint64_t>(0)) +
                   api::constant<float>(0.5040) *
                       (*input)(iter_vars[0], iter_vars[1],
                                api::constant<uint64_t>(1)) +
                   api::constant<float>(0.2570) *
                       (*input)(iter_vars[0], iter_vars[1],
                                api::constant<uint64_t>(2)) +
                   api::constant<float>(16.5),
               input->get_dtype()),
          // UV
          api::if_then_else(
              api::logical::all(condition_U),
              // U
              cast(api::constant<float>(0.4390) *
                           (*input)((iter_vars[0] - Y_h) *
                                        api::constant<uint64_t>(2),
                                    iter_vars[1], api::constant<uint64_t>(0)) +
                       api::constant<float>(-0.2910) *
                           (*input)((iter_vars[0] - Y_h) *
                                        api::constant<uint64_t>(2),
                                    iter_vars[1], api::constant<uint64_t>(1)) +
                       api::constant<float>(-0.1480) *
                           (*input)((iter_vars[0] - Y_h) *
                                        api::constant<uint64_t>(2),
                                    iter_vars[1], api::constant<uint64_t>(2)) +
                       api::constant<float>(128.5),
                   input->get_dtype()),
              // V
              cast(api::constant<float>(-0.0710) *
                           (*input)((iter_vars[0] - Y_h) *
                                        api::constant<uint64_t>(2),
                                    iter_vars[1], api::constant<uint64_t>(0)) +
                       api::constant<float>(-0.3680) *
                           (*input)((iter_vars[0] - Y_h) *
                                        api::constant<uint64_t>(2),
                                    iter_vars[1], api::constant<uint64_t>(1)) +
                       api::constant<float>(0.4390) *
                           (*input)((iter_vars[0] - Y_h) *
                                        api::constant<uint64_t>(2),
                                    iter_vars[1], api::constant<uint64_t>(2)) +
                       api::constant<float>(128.5),
                   input->get_dtype()))),
      name);
}

ir::TensorVarPtr RGB2NV12(const std::vector<ir::ExprPtr> &shape,
                          ir::TensorVarPtr input,
                          const std::string &name = "rgb2nv12") {
  ELENA_ASSERT(ptr_cast<Const<uint64_t>>(shape[2])->get_value() == 1,
               "rgb2nv12");
  ELENA_ASSERT(
      ptr_cast<Const<uint64_t>>(input->shape->element[2])->get_value() == 3,
      "rgb2nv12");

  auto iter_vars = api::construct_indices(shape);

  auto Y_h =
      cast(shape[0] * api::constant<uint64_t>(2) / api::constant<uint64_t>(3),
           input->get_dtype());
  auto Y_w =
      cast(shape[1] * api::constant<uint64_t>(2) / api::constant<uint64_t>(3),
           input->get_dtype());
  std::vector<ir::ExprPtr> condition_Y;
  condition_Y.push_back(api::logical::lt(iter_vars[0], Y_h));
  condition_Y.push_back(api::logical::lt(iter_vars[1], Y_w));

  std::vector<ir::ExprPtr> condition_U;
  condition_U.push_back(api::logical::eq(
      iter_vars[1] % api::constant<uint64_t>(2), api::constant<uint64_t>(0)));

  return api::compute(
      shape, iter_vars,
      api::if_then_else(
          api::logical::all(condition_Y),
          // Y
          cast(api::constant<float>(0.0980) *
                       (*input)(iter_vars[0], iter_vars[1],
                                api::constant<uint64_t>(2)) +
                   api::constant<float>(0.5040) *
                       (*input)(iter_vars[0], iter_vars[1],
                                api::constant<uint64_t>(1)) +
                   api::constant<float>(0.2570) *
                       (*input)(iter_vars[0], iter_vars[1],
                                api::constant<uint64_t>(0)) +
                   api::constant<float>(16.5),
               input->get_dtype()),
          // UV
          api::if_then_else(
              api::logical::all(condition_U),
              // U
              cast(api::constant<float>(0.4390) *
                           (*input)((iter_vars[0] - Y_h) *
                                        api::constant<uint64_t>(2),
                                    iter_vars[1], api::constant<uint64_t>(2)) +
                       api::constant<float>(-0.2910) *
                           (*input)((iter_vars[0] - Y_h) *
                                        api::constant<uint64_t>(2),
                                    iter_vars[1], api::constant<uint64_t>(1)) +
                       api::constant<float>(-0.1480) *
                           (*input)((iter_vars[0] - Y_h) *
                                        api::constant<uint64_t>(2),
                                    iter_vars[1], api::constant<uint64_t>(0)) +
                       api::constant<float>(128.5),
                   input->get_dtype()),
              // V
              cast(api::constant<float>(-0.0710) *
                           (*input)((iter_vars[0] - Y_h) *
                                        api::constant<uint64_t>(2),
                                    iter_vars[1], api::constant<uint64_t>(2)) +
                       api::constant<float>(-0.3680) *
                           (*input)((iter_vars[0] - Y_h) *
                                        api::constant<uint64_t>(2),
                                    iter_vars[1], api::constant<uint64_t>(1)) +
                       api::constant<float>(0.4390) *
                           (*input)((iter_vars[0] - Y_h) *
                                        api::constant<uint64_t>(2),
                                    iter_vars[1], api::constant<uint64_t>(0)) +
                       api::constant<float>(128.5),
                   input->get_dtype()))),
      name);
}
ir::TensorVarPtr BGR2NV21(const std::vector<ir::ExprPtr> &shape,
                          ir::TensorVarPtr input,
                          const std::string &name = "rgb2nv12") {
  ELENA_ASSERT(ptr_cast<Const<uint64_t>>(shape[2])->get_value() == 1,
               "bgr2nv21");
  ELENA_ASSERT(
      ptr_cast<Const<uint64_t>>(input->shape->element[2])->get_value() == 3,
      "bgr2nv21");

  auto iter_vars = api::construct_indices(shape);

  auto Y_h =
      cast(shape[0] * api::constant<uint64_t>(2) / api::constant<uint64_t>(3),
           input->get_dtype());
  auto Y_w =
      cast(shape[1] * api::constant<uint64_t>(2) / api::constant<uint64_t>(3),
           input->get_dtype());
  std::vector<ir::ExprPtr> condition_Y;
  condition_Y.push_back(api::logical::lt(iter_vars[0], Y_h));
  condition_Y.push_back(api::logical::lt(iter_vars[1], Y_w));

  std::vector<ir::ExprPtr> condition_U;
  condition_U.push_back(api::logical::ne(
      iter_vars[1] % api::constant<uint64_t>(2), api::constant<uint64_t>(0)));

  return api::compute(
      shape, iter_vars,
      api::if_then_else(
          api::logical::all(condition_Y),
          // Y
          cast(api::constant<float>(0.0980) *
                       (*input)(iter_vars[0], iter_vars[1],
                                api::constant<uint64_t>(0)) +
                   api::constant<float>(0.5040) *
                       (*input)(iter_vars[0], iter_vars[1],
                                api::constant<uint64_t>(1)) +
                   api::constant<float>(0.2570) *
                       (*input)(iter_vars[0], iter_vars[1],
                                api::constant<uint64_t>(2)) +
                   api::constant<float>(16.5),
               input->get_dtype()),
          // UV
          api::if_then_else(
              api::logical::all(condition_U),
              // U
              cast(api::constant<float>(0.4390) *
                           (*input)((iter_vars[0] - Y_h) *
                                        api::constant<uint64_t>(2),
                                    iter_vars[1], api::constant<uint64_t>(0)) +
                       api::constant<float>(-0.2910) *
                           (*input)((iter_vars[0] - Y_h) *
                                        api::constant<uint64_t>(2),
                                    iter_vars[1], api::constant<uint64_t>(1)) +
                       api::constant<float>(-0.1480) *
                           (*input)((iter_vars[0] - Y_h) *
                                        api::constant<uint64_t>(2),
                                    iter_vars[1], api::constant<uint64_t>(2)) +
                       api::constant<float>(128.5),
                   input->get_dtype()),
              // V
              cast(api::constant<float>(-0.0710) *
                           (*input)((iter_vars[0] - Y_h) *
                                        api::constant<uint64_t>(2),
                                    iter_vars[1], api::constant<uint64_t>(0)) +
                       api::constant<float>(-0.3680) *
                           (*input)((iter_vars[0] - Y_h) *
                                        api::constant<uint64_t>(2),
                                    iter_vars[1], api::constant<uint64_t>(1)) +
                       api::constant<float>(0.4390) *
                           (*input)((iter_vars[0] - Y_h) *
                                        api::constant<uint64_t>(2),
                                    iter_vars[1], api::constant<uint64_t>(2)) +
                       api::constant<float>(128.5),
                   input->get_dtype()))),
      name);
}

ir::TensorVarPtr RGB2NV21(const std::vector<ir::ExprPtr> &shape,
                          ir::TensorVarPtr input,
                          const std::string &name = "rgb2nv12") {
  ELENA_ASSERT(ptr_cast<Const<uint64_t>>(shape[2])->get_value() == 1,
               "rgb2nv21");
  ELENA_ASSERT(
      ptr_cast<Const<uint64_t>>(input->shape->element[2])->get_value() == 3,
      "rgb2nv21");

  auto iter_vars = api::construct_indices(shape);

  auto Y_h =
      cast(shape[0] * api::constant<uint64_t>(2) / api::constant<uint64_t>(3),
           input->get_dtype());
  auto Y_w =
      cast(shape[1] * api::constant<uint64_t>(2) / api::constant<uint64_t>(3),
           input->get_dtype());
  std::vector<ir::ExprPtr> condition_Y;
  condition_Y.push_back(api::logical::lt(iter_vars[0], Y_h));
  condition_Y.push_back(api::logical::lt(iter_vars[1], Y_w));

  std::vector<ir::ExprPtr> condition_U;
  condition_U.push_back(api::logical::ne(
      iter_vars[1] % api::constant<uint64_t>(2), api::constant<uint64_t>(0)));

  return api::compute(
      shape, iter_vars,
      api::if_then_else(
          api::logical::all(condition_Y),
          // Y
          cast(api::constant<float>(0.0980) *
                       (*input)(iter_vars[0], iter_vars[1],
                                api::constant<uint64_t>(2)) +
                   api::constant<float>(0.5040) *
                       (*input)(iter_vars[0], iter_vars[1],
                                api::constant<uint64_t>(1)) +
                   api::constant<float>(0.2570) *
                       (*input)(iter_vars[0], iter_vars[1],
                                api::constant<uint64_t>(0)) +
                   api::constant<float>(16.5),
               input->get_dtype()),
          // UV
          api::if_then_else(
              api::logical::all(condition_U),
              // U
              cast(api::constant<float>(0.4390) *
                           (*input)((iter_vars[0] - Y_h) *
                                        api::constant<uint64_t>(2),
                                    iter_vars[1], api::constant<uint64_t>(2)) +
                       api::constant<float>(-0.2910) *
                           (*input)((iter_vars[0] - Y_h) *
                                        api::constant<uint64_t>(2),
                                    iter_vars[1], api::constant<uint64_t>(1)) +
                       api::constant<float>(-0.1480) *
                           (*input)((iter_vars[0] - Y_h) *
                                        api::constant<uint64_t>(2),
                                    iter_vars[1], api::constant<uint64_t>(0)) +
                       api::constant<float>(128.5),
                   input->get_dtype()),
              // V
              cast(api::constant<float>(-0.0710) *
                           (*input)((iter_vars[0] - Y_h) *
                                        api::constant<uint64_t>(2),
                                    iter_vars[1], api::constant<uint64_t>(2)) +
                       api::constant<float>(-0.3680) *
                           (*input)((iter_vars[0] - Y_h) *
                                        api::constant<uint64_t>(2),
                                    iter_vars[1], api::constant<uint64_t>(1)) +
                       api::constant<float>(0.4390) *
                           (*input)((iter_vars[0] - Y_h) *
                                        api::constant<uint64_t>(2),
                                    iter_vars[1], api::constant<uint64_t>(0)) +
                       api::constant<float>(128.5),
                   input->get_dtype()))),
      name);
}

ir::TensorVarPtr NV122BGR(const std::vector<ir::ExprPtr> &shape,
                          ir::TensorVarPtr input,
                          ir::TensorVarPtr NV2BGR_params,
                          const std::string &name = "nv122bgr") {
  ELENA_ASSERT(ptr_cast<Const<uint64_t>>(shape[2])->get_value() == 3,
               "nv122bgr");
  ELENA_ASSERT(
      ptr_cast<Const<uint64_t>>(input->shape->element[2])->get_value() == 1,
      "nv122bgr");

  auto iter_vars = api::construct_indices(shape);

  return api::compute(
      shape, iter_vars,
      (*NV2BGR_params)(iter_vars[2], api::constant<uint64_t>(0)) *
              ((*input)(iter_vars[0], iter_vars[1],
                        api::constant<uint64_t>(0)) -
               api::constant<float>(16)) +  // Y
          (*NV2BGR_params)(iter_vars[2], api::constant<uint64_t>(1)) *
              ((*input)(iter_vars[0] / api::constant<uint64_t>(2) + shape[0],
                        iter_vars[1] / api::constant<uint64_t>(2) *
                            api::constant<uint64_t>(2),
                        api::constant<uint64_t>(0)) -
               api::constant<float>(128)) +  // U
          (*NV2BGR_params)(iter_vars[2], api::constant<uint64_t>(2)) *
              ((*input)(iter_vars[0] / api::constant<uint64_t>(2) + shape[0],
                        iter_vars[1] / api::constant<uint64_t>(2) *
                                api::constant<uint64_t>(2) +
                            api::constant<uint64_t>(1),
                        api::constant<uint64_t>(0)) -
               api::constant<float>(128)),  // V
      name);
}

ir::TensorVarPtr NV212BGR(const std::vector<ir::ExprPtr> &shape,
                          ir::TensorVarPtr input,
                          ir::TensorVarPtr NV2BGR_params,
                          const std::string &name = "nv212bgr") {
  ELENA_ASSERT(ptr_cast<Const<uint64_t>>(shape[2])->get_value() == 3,
               "nv212bgr");
  ELENA_ASSERT(
      ptr_cast<Const<uint64_t>>(input->shape->element[2])->get_value() == 1,
      "nv212bgr");

  auto iter_vars = api::construct_indices(shape);

  return api::compute(
      shape, iter_vars,
      (*NV2BGR_params)(iter_vars[2], api::constant<int>(0)) *
              ((*input)(iter_vars[0], iter_vars[1],
                        api::constant<uint64_t>(0)) -
               api::constant<float>(16)) +  // Y
          (*NV2BGR_params)(iter_vars[2], api::constant<int>(1)) *
              ((*input)(iter_vars[0] / api::constant<uint64_t>(2) + shape[0],
                        iter_vars[1] / api::constant<uint64_t>(2) *
                                api::constant<uint64_t>(2) +
                            api::constant<uint64_t>(1),
                        api::constant<uint64_t>(0)) -
               api::constant<float>(128)) +  // U
          (*NV2BGR_params)(iter_vars[2], api::constant<int>(2)) *
              ((*input)(iter_vars[0] / api::constant<uint64_t>(2) + shape[0],
                        iter_vars[1] / api::constant<uint64_t>(2) *
                            api::constant<uint64_t>(2),
                        api::constant<uint64_t>(0)) -
               api::constant<float>(128)),  // V
      name);
}

ir::TensorVarPtr NV122GRAY(const std::vector<ir::ExprPtr> &shape,
                           ir::TensorVarPtr input,
                           const std::string &name = "nv122gray") {
  ELENA_ASSERT(ptr_cast<Const<uint64_t>>(shape[2])->get_value() == 1,
               "nv122gray");
  ELENA_ASSERT(
      ptr_cast<Const<uint64_t>>(input->shape->element[2])->get_value() == 1,
      "nv122gray");

  auto iter_vars = api::construct_indices(shape);

  return api::compute(
      shape, iter_vars,
      (*input)(iter_vars[0], iter_vars[1], api::constant<uint64_t>(0)), name);
}

ir::TensorVarPtr NV212GRAY(const std::vector<ir::ExprPtr> &shape,
                           ir::TensorVarPtr input,
                           const std::string &name = "nv212gray") {
  ELENA_ASSERT(ptr_cast<Const<uint64_t>>(shape[2])->get_value() == 1,
               "nv212gray");
  ELENA_ASSERT(
      ptr_cast<Const<uint64_t>>(input->shape->element[2])->get_value() == 1,
      "nv212gray");

  auto iter_vars = api::construct_indices(shape);

  return api::compute(
      shape, iter_vars,
      (*input)(iter_vars[0], iter_vars[1], api::constant<uint64_t>(0)), name);
}

}  // namespace CvtColor
