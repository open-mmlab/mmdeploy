#include <iostream>
#include <string>

#include "Crop.hpp"
#include "CvtColor.hpp"
#include "Json.hpp"
#include "LayoutTrans.hpp"
#include "Norm.hpp"
#include "Pad.hpp"
#include "Resize.hpp"

#define BLOCK_SIZE 128  // for cuda device

int main(int argc, char *argv[]) {
  if (argc < 4) {
    ELENA_WARN(
        "usage: OpFuse <path/of/OpList/json/file> <cpu or cuda> "
        "<path/of/generate/code>");
  }

  auto json_path = (std::string)argv[1];
  auto device = (std::string)argv[2];

  auto write_path = (std::string) "./";
  if (argc == 4) write_path = (std::string)argv[3];

  if (device != "cpu" && device != "cuda") {
    ELENA_ABORT("device only support <cpu or cuda>");
  }

  /* 1. json parsing module */
  std::vector<std::string> OpList;
  Format CvtFormat;
  readOpList(json_path, OpList, CvtFormat);

  bool ResizeOp =
      std::find(OpList.begin(), OpList.end(), "Resize") != OpList.end() ? true
                                                                        : false;

  Target target = device == "cpu" ? CPU : CUDA;
  Dtype dtype = Uint8;  // also support float32
  std::ostringstream gen_code;
  gen_code << Common::prelude;
  if (device == "cuda")
    gen_code << Common::cuda_prelude << BLOCK_SIZE << std::endl;

  /* 2. Traverse diff Format generate different kernel */
  for (int cur_format = 1; cur_format <= 6; cur_format++) {
    for (int interpolation = 1; interpolation <= (ResizeOp ? 2 : 1);
         interpolation++) {
      Format format = (Format)cur_format;
      Interpolation ResizeInterpolation =
          ResizeOp ? (Interpolation)interpolation
                   : Nearest;  // if no ResizeOp, use nearest replace

      /* 2.1 Initialization */
      auto h = std::make_shared<ir::ScalarVar>("h", ir::ScalarType::UInt64);
      auto w = std::make_shared<ir::ScalarVar>("w", ir::ScalarType::UInt64);
      ir::ExprPtr c;
      ir::TensorVarPtr img;
      Common::GenerateChannelAndInput(format, dtype, h, w, c, img);

      /* 2.2 OpList to IR Stage */
      auto intermediate = img;

      std::vector<ir::TensorVarPtr> stage_list;
      ir::Array<ir::IterVar> iter_vars;
      std::vector<ir::ExprPtr> arg_list;

      // pin the arglist
      auto resize_h =
          std::make_shared<ir::ScalarVar>("resize_h", ir::ScalarType::UInt64);
      auto resize_w =
          std::make_shared<ir::ScalarVar>("resize_w", ir::ScalarType::UInt64);
      auto crop_size =
          std::make_shared<ir::ScalarVar>("crop_size", ir::ScalarType::UInt64);
      auto crop_top =
          std::make_shared<ir::ScalarVar>("crop_top", ir::ScalarType::Int32);
      auto crop_left =
          std::make_shared<ir::ScalarVar>("crop_left", ir::ScalarType::Int32);
      auto norm_mean_0 = std::make_shared<ir::ScalarVar>(
          "norm_mean_0", ir::ScalarType::Float32);
      auto norm_mean_1 = std::make_shared<ir::ScalarVar>(
          "norm_mean_1", ir::ScalarType::Float32);
      auto norm_mean_2 = std::make_shared<ir::ScalarVar>(
          "norm_mean_2", ir::ScalarType::Float32);
      auto norm_std_0 = std::make_shared<ir::ScalarVar>(
          "norm_std_0", ir::ScalarType::Float32);
      auto norm_std_1 = std::make_shared<ir::ScalarVar>(
          "norm_std_1", ir::ScalarType::Float32);
      auto norm_std_2 = std::make_shared<ir::ScalarVar>(
          "norm_std_2", ir::ScalarType::Float32);
      auto pad_h =
          std::make_shared<ir::ScalarVar>("pad_h", ir::ScalarType::UInt64);
      auto pad_w =
          std::make_shared<ir::ScalarVar>("pad_w", ir::ScalarType::UInt64);
      auto pad_top =
          std::make_shared<ir::ScalarVar>("pad_top", ir::ScalarType::Int32);
      auto pad_left =
          std::make_shared<ir::ScalarVar>("pad_left", ir::ScalarType::Int32);
      auto pad_bottom =
          std::make_shared<ir::ScalarVar>("pad_bottom", ir::ScalarType::Int32);
      auto pad_right =
          std::make_shared<ir::ScalarVar>("pad_right", ir::ScalarType::Int32);
      auto pad_value =
          std::make_shared<ir::ScalarVar>("pad_value", ir::ScalarType::Float32);

      arg_list.push_back(resize_h);
      arg_list.push_back(resize_w);
      arg_list.push_back(crop_size);
      arg_list.push_back(crop_top);
      arg_list.push_back(crop_left);
      arg_list.push_back(norm_mean_0);
      arg_list.push_back(norm_mean_1);
      arg_list.push_back(norm_mean_2);
      arg_list.push_back(norm_std_0);
      arg_list.push_back(norm_std_1);
      arg_list.push_back(norm_std_2);
      arg_list.push_back(pad_h);
      arg_list.push_back(pad_w);
      arg_list.push_back(pad_top);
      arg_list.push_back(pad_left);
      arg_list.push_back(pad_bottom);
      arg_list.push_back(pad_right);
      arg_list.push_back(pad_value);

      auto two = api::constant<uint64_t>(2);
      auto cubfh = api::placeholder<int16_t>({two, resize_h}, "cubfh");
      auto cubfw = api::placeholder<int16_t>({two, resize_w}, "cubfw");
      auto inth = api::placeholder<int32_t>({two, resize_h}, "inth");
      auto intw = api::placeholder<int32_t>({two, resize_w}, "intw");

      /* common expression extraction */
      ir::TensorVarPtr mean_value;
      ir::TensorVarPtr std_value;
      ir::TensorVarPtr nv2bgr_params = Common::NV2BGRParams();

      if (std::find(OpList.begin(), OpList.end(), "Normalize") !=
          OpList.end()) {
        std::vector<ir::ExprPtr> mean_vec{norm_mean_0, norm_mean_1,
                                          norm_mean_2};
        std::vector<ir::ExprPtr> std_vec{norm_std_0, norm_std_1, norm_std_2};
        mean_value = CvtFormat == BGR ? Common::BGRMean(mean_vec)
                                      : Common::GrayMean(mean_vec);
        std_value = CvtFormat == BGR ? Common::BGRStd(std_vec)
                                     : Common::GrayStd(std_vec);
      }

      for (auto &op : OpList) {
        ir::TensorVarPtr cur_stage;
        if (op == "cvtColorBGR" || op == "cvtColorGRAY") {
          if (CvtFormat == BGR) {
            std::vector<ir::ExprPtr> color_shape{h, w,
                                                 api::constant<uint64_t>(3)};
            if (format == BGR) {
              continue;
            } else if (format == RGB) {
              cur_stage = CvtColor::RGB2BGR(color_shape, intermediate);
            } else if (format == GRAY) {
              cur_stage = CvtColor::GRAY2BGR(color_shape, intermediate);
            } else if (format == BGRA) {
              cur_stage = CvtColor::BGRA2BGR(color_shape, intermediate);
            } else if (format == NV12) {
              cur_stage =
                  CvtColor::NV122BGR(color_shape, intermediate, nv2bgr_params);
            } else if (format == NV21) {
              cur_stage =
                  CvtColor::NV212BGR(color_shape, intermediate, nv2bgr_params);
            } else {
              ELENA_ABORT(
                  "not support temporarily in CvtColr when CvtFormat == BGR");
            }
          } else if (CvtFormat == GRAY) {
            std::vector<ir::ExprPtr> color_shape{h, w,
                                                 api::constant<uint64_t>(1)};
            if (format == BGR) {
              cur_stage = CvtColor::BGR2GRAY(color_shape, intermediate);
            } else if (format == RGB) {
              cur_stage = CvtColor::RGB2GRAY(color_shape, intermediate);
            } else if (format == GRAY) {
              continue;
            } else if (format == BGRA) {
              cur_stage = CvtColor::BGRA2GRAY(color_shape, intermediate);
            } else if (format == NV12) {
              cur_stage = CvtColor::NV122GRAY(color_shape, intermediate);
            } else if (format == NV21) {
              cur_stage = CvtColor::NV212GRAY(color_shape, intermediate);
            } else {
              ELENA_ABORT(
                  "not support temporarily in CvtColr when CvtFormat == GRAY");
            }
          } else {
            ELENA_ABORT("not support temporarily in CvtColr");
          }
        } else if (op == "Resize") {
          std::vector<ir::ExprPtr> resize_shape{
              resize_h, resize_w, intermediate->shape->element[2]};
          iter_vars = api::construct_indices(resize_shape);
          if (ResizeInterpolation == Nearest) {
            cur_stage = Resize::Nearest(resize_shape, iter_vars, intermediate);
          } else if (ResizeInterpolation == Bilinear) {
            /* aim to uint8 condition */
            arg_list.push_back(cubfh);
            arg_list.push_back(cubfw);
            arg_list.push_back(inth);
            arg_list.push_back(intw);
            cur_stage = Resize::Bilinear(resize_shape, iter_vars, intermediate,
                                         cubfh, cubfw, inth, intw);
          } else {
            ELENA_ABORT("not support temporarily in Resize");
          }
        } else if (op == "CenterCrop") {
          std::vector<ir::ExprPtr> crop_shape{crop_size, crop_size,
                                              intermediate->shape->element[2]};
          iter_vars = api::construct_indices(crop_shape);
          cur_stage = Crop::Crop(crop_shape, iter_vars, intermediate, crop_top,
                                 crop_left);
        } else if (op == "CastFloat") {
          std::vector<ir::ExprPtr> cast_shape{intermediate->shape->element[0],
                                              intermediate->shape->element[1],
                                              intermediate->shape->element[2]};
          iter_vars = api::construct_indices(cast_shape);
          cur_stage = Common::CastFloat(cast_shape, iter_vars, intermediate);
        } else if (op == "cvtColorRGB") {
          ELENA_ASSERT(CvtFormat == BGR, "");
          std::vector<ir::ExprPtr> rgb_shape{intermediate->shape->element[0],
                                             intermediate->shape->element[1],
                                             intermediate->shape->element[2]};
          iter_vars = api::construct_indices(rgb_shape);
          cur_stage = CvtColor::BGR2RGB(rgb_shape, iter_vars, intermediate);
        } else if (op == "Normalize") {
          std::vector<ir::ExprPtr> norm_shape{intermediate->shape->element[0],
                                              intermediate->shape->element[1],
                                              intermediate->shape->element[2]};
          iter_vars = api::construct_indices(norm_shape);
          cur_stage = Norm::FloatNorm(norm_shape, iter_vars, intermediate,
                                      mean_value, std_value);
        } else if (op == "Pad") {
          std::vector<ir::ExprPtr> padding_tlbr{pad_top, pad_left, pad_bottom,
                                                pad_right};
          std::vector<ir::ExprPtr> pad_shape{pad_h, pad_w,
                                             intermediate->shape->element[2]};
          iter_vars = api::construct_indices(pad_shape);
          cur_stage = Pad::Pad(pad_shape, iter_vars, intermediate, padding_tlbr,
                               pad_value);
        } else if (op == "HWC2CHW") {
          std::vector<ir::ExprPtr> trans_shape{intermediate->shape->element[2],
                                               intermediate->shape->element[0],
                                               intermediate->shape->element[1]};
          std::vector<ir::ExprPtr> trans_iter{intermediate->shape->element[0],
                                              intermediate->shape->element[1],
                                              intermediate->shape->element[2]};
          iter_vars = api::construct_indices(trans_iter);
          cur_stage =
              LayoutTrans::HWC2CHW(trans_shape, iter_vars, intermediate);
        } else {
          ELENA_ABORT("not support this " << op << " op temporarily");
        }

        stage_list.push_back(cur_stage);
        intermediate = cur_stage;
      }

      auto sch = api::create_schedule(intermediate->op);

      std::string final_op = stage_list[stage_list.size() - 1]->get_name();
      for (auto stage : stage_list) {
        if (stage->get_name() != final_op) {
          (*sch)[stage]->compute_inline();
        }
      }

      if (target == CUDA) {
        auto fuse = (*sch)[intermediate->op]->fuse(
            {iter_vars[0], iter_vars[1], iter_vars[2]});
        auto t = (*sch)[intermediate->op]->split(
            fuse, api::constant<uint64_t>(BLOCK_SIZE));
        (*sch)[intermediate->op]->set_bind(t.element[0], "blockIdx.x");
        (*sch)[intermediate->op]->set_bind(t.element[1], "threadIdx.x");
      }

      sch = sch->normalize();
      auto bound_map = api::inferBound(sch);
      auto stmt = api::scheduleToStatement(sch, bound_map);
      stmt = api::flattenStorage(stmt, bound_map);
      stmt = api::rewriteStorage(stmt);
      stmt = api::autoUnroll(stmt, false);
      // stmt = api::simplify(stmt);
      // stmt = api::simplifyStatement(stmt);

      std::string kernel_name;
      if (format == BGR)
        kernel_name += "BGR_";
      else if (format == GRAY)
        kernel_name += "GRAY_";
      else if (format == RGB)
        kernel_name += "RGB_";
      else if (format == BGRA)
        kernel_name += "BGRA_";
      else if (format == NV12)
        kernel_name += "NV12_";
      else if (format == NV21)
        kernel_name += "NV21_";

      if (ResizeInterpolation == Nearest)
        kernel_name += "Nearest_";
      else if (ResizeInterpolation == Bilinear)
        kernel_name += "Bilinear_";

      // if(dtype == Uint8)
      //   kernel_name += "Uint8_";
      // else if(dtype == Float32)
      //   kernel_name += "Float32_";

      kernel_name += "Kernel";

      arg_list.push_back(img);
      arg_list.push_back(intermediate);
      arg_list.push_back(h);
      arg_list.push_back(w);

      if (target == CUDA) {
        std::string cuda_code_row =
            api::genCudaSrc(stmt, arg_list, kernel_name);
        std::string cuda_code = cuda_code_row;
        gen_code << cuda_code;
      } else if (target == CPU) {
        std::string x86_code_row = api::genX86Src(stmt, arg_list, kernel_name);
        std::string x86_code = x86_code_row;
        gen_code << x86_code;
      }
    }
  }

  auto cc = write_path + (target == CPU ? "/source.c" : "/source.cu");

  if (target == CUDA) {
    if (ResizeOp)
      gen_code << Common::cuda_bilinear_preprocess_func
               << Common::cuda_call_func_begin << Common::cuda_bilinear_func
               << Common::call_func_end;
    else
      gen_code << Common::cuda_call_func_begin << Common::call_func_end;
    api::dump_code(gen_code.str(), cc);
  } else {
    if (ResizeOp)
      gen_code << Common::cpu_bilinear_preprocess_func
               << Common::cpu_call_func_begin << Common::cpu_bilinear_func
               << Common::call_func_end;
    else
      gen_code << Common::cpu_call_func_begin << Common::call_func_end;
    api::dump_code(gen_code.str(), cc);
  }

  return 0;
}
