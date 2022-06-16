// Copyright (c) OpenMMLab. All rights reserved.

#include "shape_inference.h"

#include <algorithm>

/**
 * @brief query output shape of target node
 *
 * @param mutable_graph
 * @param target
 * @param weights
 * @param context <tensor name, shape>
 * @return std::tuple<bool, std::vector<int>>
 */
std::tuple<bool, std::vector<int>> query_shape(
    onnx::GraphProto* mutable_graph, onnx::NodeProto* target,
    const std::map<std::string, onnx::TensorProto>& weights,
    std::map<std::string, std::vector<int>>& context) {
  // emplace all input nodes
  const int input_count = mutable_graph->input_size();
  for (int i = 0; i < input_count; i++) {
    auto inp = mutable_graph->input(i);
    onnx::TypeProto inp_type = inp.type();
    onnx::TensorShapeProto shape_proto = inp_type.tensor_type().shape();

    auto dim_size = shape_proto.dim_size();
    std::vector<int> shape(dim_size);
    for (int index = 0; index < dim_size; ++index) {
      shape[index] = shape_proto.dim(index).dim_value();
    }

    context.emplace(inp.name(), shape);
  }

  // BFS the tree, `target` as root, onnx::graph inputs and weights as leaf nodes
  std::vector<onnx::NodeProto*> serial = {target};
  {
    std::set<std::string> mark_as_appended = {};
    while (true) {
      int start = 0, end = serial.size();
      for (int i = start; i < end; ++i) {
        auto node_ptr = serial[i];
        auto len = node_ptr->input_size();

        for (int j = 0; j < len; ++j) {
          std::string name = node_ptr->input(j);
          if (context.find(name) != context.end()) {
            // if input founded, skip
            continue;
          }

          if (weights.find(name) != weights.end()) {
            // if founded in weights, extract shape to context
            auto weight = weights.at(name);
            std::vector<int> shape;
            for (auto index = 0; index < weight.dims_size(); ++index) {
              shape.emplace_back(weight.dims(index));
            }
            context.emplace(name, shape);
            continue;
          }

          if (mark_as_appended.find(name) != mark_as_appended.end()) {
            // if mark as appended, skip
            continue;
          }
          // else append it to serialization list
          auto depend_ptr = find_node_by_output_name(mutable_graph, name);
          if (depend_ptr == nullptr) {
            fprintf(stderr, "cannot find %s from graph !\n", name.c_str());
            return std::make_tuple(false, std::vector<int>{});
          }
          mark_as_appended.insert(name);
          serial.emplace_back(depend_ptr);
        }
      }

      if (serial.size() <= end) {
        // if not new node added, quit
        break;
      }

      // update start and end position, continue BFS the tree
      start = end;
      end = serial.size();
    }
  }

  // for each node in serialization list, calculate the output shape
  {
    std::reverse(serial.begin(), serial.end());
    for (auto node : serial) {
      if (node->op_type() == "Conv") {
        auto inp = context[node->input(0)];
        auto weight = context[node->input(1)];
        assert(inp.size() == 4 and weight.size() == 4);

        int group = get_node_attr_i(*node, "group", 1);
        assert(group == 1);

        // treat multiple spatial attr as single one
#define EXTRACT_REPEATED_PARAM(NAME, ATTR, DEFAULT)        \
  int ATTR = DEFAULT;                                      \
  {                                                        \
    std::vector<int> _vec = get_node_attr_ai(*node, NAME); \
    if (not _vec.empty()) {                                \
      ATTR = _vec[0];                                      \
    }                                                      \
  }

        EXTRACT_REPEATED_PARAM("dilations", dilation, 1);
        EXTRACT_REPEATED_PARAM("pads", pad, 0);
        EXTRACT_REPEATED_PARAM("strides", stride, 1);

#undef EXTRACT_REPEATED_PARAM

        int on = inp[0];
        int oc = weight[0];
        int oh = (inp[2] + 2 * pad - weight[2]) / stride + 1;
        int ow = (inp[3] + 2 * pad - weight[3]) / stride + 1;
        context.emplace(node->output(0), std::vector<int>{on, oc, oh, ow});

      } else if (node->op_type() == "Shape") {
        auto inp = context[node->input(0)];
        context.emplace(node->output(0), std::vector<int>{1, inp[1], inp[2], inp[3]});

      } else if (node->op_type() == "Slice") {
        assert(node->input_size() >= 4);

        auto inp = context[node->input(0)];
        int start = get_node_attr_from_input<int>(weights.at(node->input(1)));
        int end = get_node_attr_from_input<int>(weights.at(node->input(2)));
        int axes = get_node_attr_from_input<int>(weights.at(node->input(3)));

        if (axes != 0) {
          fprintf(stderr, "Not support axes=%d !\n", axes);
          return std::make_tuple(false, std::vector<int>{});
        }

        assert(inp.size() >= end - start);
        context.emplace(node->output(0), std::vector<int>{inp.begin() + start, inp.begin() + end});

      } else if (node->op_type() == "Concat") {
        assert(node->input_size() >= 2);

        auto axis = get_node_attr_i(*node, "axis", 0);
        if (axis != 0) {
          fprintf(stderr, "Not support axes=%d !\n", axis);
          return std::make_tuple(false, std::vector<int>{});
        }

        std::vector<int> inp = context[node->input(0)];
        std::vector<int> w_data = get_node_attr_from_input_ai(weights.at(node->input(1)));

        // concat data on axis 0
        inp.insert(inp.end(), w_data.begin(), w_data.end());
        context.emplace(node->output(0), inp);

      } else {
        fprintf(stderr, "Unsupported type %s in query_shape !\n", node->op_type().c_str());
        return std::make_tuple(false, std::vector<int>{});
      }
    }
  }

  assert(context.find(target->output(0)) != context.end());
  auto target_shape = context[target->output(0)];
  return std::make_tuple(true, target_shape);
}
