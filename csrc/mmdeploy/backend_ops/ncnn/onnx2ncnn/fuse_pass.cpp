// Copyright (c) OpenMMLab. All rights reserved.
#include "fuse_pass.h"

void fuse_rewrite_gather(onnx::GraphProto* mutable_graph,
                         std::map<std::string, onnx::TensorProto>& weights,
                         std::map<std::string, int>& node_reference,
                         std::set<std::string>& blob_names, int& reduced_node_count) {
  const int node_count = mutable_graph->node_size();
  for (int i = 0; i < node_count; ++i) {
    onnx::NodeProto* gather = mutable_graph->mutable_node(i);
    if (gather->op_type() != "Gather") {
      continue;
    }
    auto indices = get_node_attr_from_input_ai(weights[gather->input(1)]);
    if (indices.size() != 1) {
      continue;
    }

    {
      // reconstruct node connections
      node_reference[gather->input(1)] -= 1;
      std::string origin_inp = gather->input(0);
      gather->clear_input();
      gather->add_input(origin_inp);
    }

    {
      // update axis, starts and ends
      int axis = get_node_attr_i(*gather, "axis", 1) - 1;

      gather->set_op_type("Crop");
      gather->clear_attribute();

      int indice = indices[0];
      set_node_attr_ai(*gather, "starts", std::vector<int>{indice});
      set_node_attr_ai(*gather, "ends", std::vector<int>{indice + 1});
      set_node_attr_ai(*gather, "axis", std::vector<int>{axis});
    }
  }
}

void fuse_weight_reshape(onnx::GraphProto* mutable_graph,
                         std::map<std::string, onnx::TensorProto>& weights,
                         std::map<std::string, int>& node_reference,
                         std::set<std::string>& blob_names, int& reduced_node_count) {
  int node_count = mutable_graph->node_size();
  for (int i = 0; i < node_count; i++) {
    onnx::NodeProto* node = mutable_graph->mutable_node(i);

    // weight <= Reshape(weight)
    if (node->op_type() == "Reshape") {
      // check weight
      if (weights.find(node->input(0)) == weights.end()) continue;

      weights[node->output(0)] = weights[node->input(0)];

      // set weight shape directly
      std::vector<int> shape;
      if (node->input_size() == 1) {
        shape = get_node_attr_ai(*node, "shape");
      } else if (node->input_size() == 2) {
        // opset 5
        shape = get_node_attr_from_input_ai(weights[node->input(1)]);
      }

      weights[node->output(0)].clear_dims();
      for (int j = 0; j < shape.size(); j++) {
        weights[node->output(0)].add_dims(shape[j]);
      }

      // reduce
      node->set_op_type("noop_reducedncnn");

      node_reference[node->input(0)] -= 1;
      if (node->input_size() == 2) {
        node_reference[node->input(1)] -= 1;
      }

      reduced_node_count += 1;
      i += 1;
    }
  }
}

void fuse_weight_transpose(onnx::GraphProto* mutable_graph,
                           std::map<std::string, onnx::TensorProto>& weights,
                           std::map<std::string, int>& node_reference,
                           std::set<std::string>& blob_names, int& reduced_node_count) {
  int node_count = mutable_graph->node_size();
  for (int i = 0; i < node_count; i++) {
    onnx::NodeProto* node = mutable_graph->mutable_node(i);

    // weight <= Transpose(weight)
    if (node->op_type() == "Transpose") {
      // check weight
      if (weights.find(node->input(0)) == weights.end()) continue;

      if (weights[node->input(0)].dims_size() != 2) continue;

      // perm = (1, 0)
      std::vector<int> perm = get_node_attr_ai(*node, "perm");
      if (perm.size() != 2) continue;
      if (perm[0] != 1 || perm[1] != 0) continue;

      weights[node->output(0)] = weights[node->input(0)];

      // permute weight
      {
        onnx::TensorProto& B = weights[node->output(0)];

        const int h = B.dims(0);
        const int w = B.dims(1);

        std::vector<float> permuted_data;
        permuted_data.reserve((size_t)h * w);
        const float* bptr =
            B.has_raw_data() ? (const float*)B.raw_data().data() : B.float_data().data();

        for (int j = 0; j < w; j++) {
          for (int k = 0; k < h; k++) {
            float vb = bptr[k * w + j];
            permuted_data.push_back(vb);
          }
        }

        B.set_dims(0, w);
        B.set_dims(1, h);

        if (B.has_raw_data()) {
          B.set_raw_data(permuted_data.data(), permuted_data.size() * sizeof(float));
        } else {
          for (int j = 0; j < (int)permuted_data.size(); j++) B.set_float_data(j, permuted_data[j]);
        }
      }

      // reduce
      node->set_op_type("noop_reducedncnn");

      node_reference[node->input(0)] -= 1;

      reduced_node_count += 1;
      i += 1;
    }
  }
}

void fuse_shufflechannel(onnx::GraphProto* mutable_graph,
                         std::map<std::string, onnx::TensorProto>& weights,
                         std::map<std::string, int>& node_reference,
                         std::set<std::string>& blob_names, int& reduced_node_count) {
  int node_count = mutable_graph->node_size();
  for (int i = 0; i < node_count; i++) {
    onnx::NodeProto* node = mutable_graph->mutable_node(i);

    // ShuffleChannel <= Reshape - Transpose - Reshape
    // ShuffleChannel <= Reshape - Transpose - Constant - Reshape
    if (node->op_type() == "Reshape") {
      if (node_reference[node->output(0)] != 1) continue;

      std::vector<int> shape;
      if (node->input_size() == 1) {
        shape = get_node_attr_ai(*node, "shape");
      } else {
        // skip weight reshape
        if (weights.find(node->input(1)) == weights.end()) continue;

        shape = get_node_attr_from_input_ai(weights[node->input(1)]);
      }

      // 1 groups channels_per_group, height, width
      // reverse style = channels_per_group, groups, height * width
      if (shape.size() != 5 && shape.size() != 3) continue;

      if (shape.size() == 5 && shape[0] != 1) continue;

      if (i + 2 >= node_count) continue;

      onnx::NodeProto* node2 = mutable_graph->mutable_node(i + 1);
      onnx::NodeProto* node3 = mutable_graph->mutable_node(i + 2);

      if (node3->op_type() == "Constant") {
        if (i + 3 >= node_count) continue;

        node3 = mutable_graph->mutable_node(i + 3);
      }

      if (node2->op_type() != "Transpose" || node3->op_type() != "Reshape") continue;

      if (node_reference[node2->output(0)] != 1) continue;

      // 0 2 1 3 4
      // reverse style = 1 0 2
      std::vector<int> perm = get_node_attr_ai(*node2, "perm");
      if (perm.size() != 5 && perm.size() != 3) continue;

      if (perm.size() == 5 &&
          (perm[0] != 0 || perm[1] != 2 || perm[2] != 1 || perm[3] != 3 || perm[4] != 4))
        continue;

      if (perm.size() == 3 && (perm[0] != 1 || perm[1] != 0 || perm[2] != 2)) continue;

      std::vector<int> shape3;
      if (node3->input_size() == 1) {
        shape3 = get_node_attr_ai(*node3, "shape");
      } else {
        // skip weight reshape
        if (weights.find(node3->input(1)) == weights.end()) continue;

        shape3 = get_node_attr_from_input_ai(weights[node3->input(1)]);
      }

      // 1, -1, height, width
      // reverse style = group, -1, channels_per_group, height, width
      if (shape3.size() != 4 && shape3.size() != 5) continue;

      if (shape3.size() == 4 &&
          (shape3[0] != 1 || (shape3[1] != -1 && shape3[1] != shape[1] * shape[2])))
        continue;

      if (shape3.size() == 5 &&
          (shape3[0] != shape[1] || shape3[2] != shape[0] || shape3[3] * shape3[4] != shape[2]))
        continue;

      // reduce
      node->set_op_type("noop_reducedncnn");
      node2->set_op_type("noop_reducedncnn");

      if (node->input_size() == 2) {
        node_reference[node->input(1)] -= 1;
      }
      node_reference[node->output(0)] -= 1;
      node_reference[node2->output(0)] -= 1;
      if (node3->input_size() == 2) {
        node_reference[node3->input(1)] -= 1;
      }

      blob_names.erase(node->output(0));
      blob_names.erase(node2->output(0));

      node3->set_op_type("ShuffleChannel");
      node3->set_input(0, node->input(0));

      onnx::AttributeProto* attr_group = node3->add_attribute();
      attr_group->set_name("group");
      attr_group->set_i(shape[1]);

      onnx::AttributeProto* attr_reverse = node3->add_attribute();
      attr_reverse->set_name("reverse");
      attr_reverse->set_i(shape.size() == 3);

      reduced_node_count += 2;
      i += 2;
    }
  }
}

void fuse_shufflechannel_split(onnx::GraphProto* mutable_graph,
                               std::map<std::string, onnx::TensorProto>& weights,
                               std::map<std::string, int>& node_reference,
                               std::set<std::string>& blob_names, int& reduced_node_count) {
  int node_count = mutable_graph->node_size();
  for (int i = 0; i < node_count; i++) {
    onnx::NodeProto* node = mutable_graph->mutable_node(i);

    // Split <= ShuffleChannel(reverse type) - Gather(0) - Gather(1)
    if (node->op_type() == "ShuffleChannel") {
      // reverse = 1
      int reverse = get_node_attr_i(*node, "reverse");
      if (reverse != 1) continue;

      if (i + 2 >= node_count) continue;

      onnx::NodeProto* node2 = mutable_graph->mutable_node(i + 1);
      onnx::NodeProto* node3 = mutable_graph->mutable_node(i + 2);

      if (node2->op_type() != "Gather" || node3->op_type() != "Gather") continue;

      if (node2->input(0) != node->output(0) || node3->input(0) != node->output(0)) continue;

      // axis = 0
      int gather2_axis = get_node_attr_i(*node2, "axis");
      if (gather2_axis != 0) continue;

      // indices = 0
      if (weights.find(node2->input(1)) == weights.end()) continue;

      std::vector<int> gather2_indices = get_node_attr_from_input_ai(weights[node2->input(1)]);
      if (gather2_indices.size() != 1 || gather2_indices[0] != 0) continue;

      // axis = 0
      int gather3_axis = get_node_attr_i(*node3, "axis");
      if (gather3_axis != 0) continue;

      // indices = 1
      if (weights.find(node3->input(1)) == weights.end()) continue;

      std::vector<int> gather3_indices = get_node_attr_from_input_ai(weights[node3->input(1)]);
      if (gather3_indices.size() != 1 || gather3_indices[0] != 1) continue;

      // reduce
      node2->set_op_type("noop_reducedncnn");

      node_reference[node->output(0)] -= 2;
      node_reference[node2->input(1)] -= 1;
      node_reference[node3->input(1)] -= 1;

      node3->set_op_type("Split");
      node3->clear_input();
      node3->add_input(node->output(0));
      node3->add_output(node3->output(0));
      node3->set_output(0, node2->output(0));

      node3->clear_attribute();
      onnx::AttributeProto* attr_axis = node3->add_attribute();
      attr_axis->set_name("axis");
      attr_axis->set_i(1);

      reduced_node_count += 1;
      i += 1;
    }
  }
}

/**
 * @brief fuse subgraph
 *
 * conv - - - - - - - - - - - -> reshape
 *     \                        /
 *       shape - slice - concat
 *
 * to
 *
 * conv --> reshape
 *
 * @param mutable_graph
 * @param weights
 * @param node_reference
 * @param blob_names
 * @param reduced_node_count
 */
void fuse_conv_reshape(onnx::GraphProto* mutable_graph,
                       std::map<std::string, onnx::TensorProto>& weights,
                       std::map<std::string, int>& node_reference,
                       std::set<std::string>& blob_names, int& reduced_node_count) {
  std::map<std::string, std::vector<int>> shape_context;
  const int node_count = mutable_graph->node_size();

  for (int i = 0; i < node_count; i++) {
    onnx::NodeProto* conv = mutable_graph->mutable_node(i);

    if (conv->op_type() != "Conv") {
      continue;
    }

    if (i + 4 >= node_count) {
      continue;
    }

    onnx::NodeProto *shape = nullptr, *slice = nullptr, *concat = nullptr, *reshape = nullptr;

    // match [Shape ... Slice, Concat ... Reshape] from near sequence, skip useless Constant
    std::vector<std::tuple<std::string, onnx::NodeProto**>> candidates = {
        {"Shape", &shape}, {"Slice", &slice}, {"Concat", &concat}, {"Reshape", &reshape}};

    int MAX = std::min(10, node_count - i - 1);
    int pos_candidate = 0;

    for (int j = 0; j < MAX; ++j) {
      auto node_ptr = mutable_graph->mutable_node(j + i + 1);
      if (node_ptr->op_type() == "Constant") {
        continue;
      }
      if (node_ptr->op_type() == std::get<0>(candidates[pos_candidate])) {
        *(std::get<1>(candidates[pos_candidate])) = node_ptr;
        pos_candidate++;
      }
    }

    if (pos_candidate != candidates.size()) {
      // not match the sequence
      continue;
    }

    if (node_reference[conv->output(0)] != 2 || node_reference[shape->output(0)] != 1 ||
        node_reference[slice->output(0)] != 1 || node_reference[concat->output(0)] != 1 ||
        node_reference[reshape->output(0)] != 1) {
      continue;
    }

    // check the connections
    if (shape->input(0) != conv->output(0) || reshape->input(0) != conv->output(0)) {
      continue;
    }
    if (slice->input(0) != shape->output(0)) {
      continue;
    }
    if (concat->input(0) != slice->output(0)) {
      continue;
    }
    if (reshape->input(0) != conv->output(0) || reshape->input(1) != concat->output(0)) {
      continue;
    }

    // add reshape attr
    auto result = query_shape(mutable_graph, concat, weights, shape_context);
    if (!std::get<0>(result)) {
      continue;
    }
    set_node_attr_ai(*reshape, "shape", std::get<1>(result));

    // reconstruct graph
    {
      // remove reference
      node_reference[reshape->input(1)] -= 1;
      node_reference[concat->input(0)] -= 1;
      node_reference[slice->input(0)] -= 1;
      node_reference[shape->input(0)] -= 1;

      // remove tensor/blob on edge
      blob_names.erase(slice->input(0));
      blob_names.erase(slice->input(1));
      blob_names.erase(slice->input(2));
      blob_names.erase(slice->input(3));
      weights.erase(slice->input(1));
      weights.erase(slice->input(2));
      weights.erase(slice->input(3));

      blob_names.erase(concat->input(0));
      blob_names.erase(concat->input(1));
      weights.erase(concat->input(1));

      blob_names.erase(reshape->input(0));

      // update edge
      shape->clear_input();
      reshape->clear_input();
      reshape->add_input(conv->output(0));

      shape->set_op_type("noop_reducedncnn");
      slice->set_op_type("noop_reducedncnn");
      concat->set_op_type("noop_reducedncnn");

      reduced_node_count += 3;
    }
    i += 3;
  }
}

void fuse_binaryop_with_scalar(onnx::GraphProto* mutable_graph,
                               std::map<std::string, onnx::TensorProto>& weights,
                               std::map<std::string, int>& node_reference,
                               std::set<std::string>& blob_names, int& reduced_node_count) {
  int node_count = mutable_graph->node_size();
  for (int i = 0; i < node_count; i++) {
    onnx::NodeProto* node = mutable_graph->mutable_node(i);

    // Add/Sub/Mul/Div/Min/Max/Pow
    if (node->op_type() == "Add" || node->op_type() == "Sub" || node->op_type() == "Mul" ||
        node->op_type() == "Div" || node->op_type() == "Max" || node->op_type() == "Min" ||
        node->op_type() == "Pow") {
      if (weights.find(node->input(1)) == weights.end()) continue;

      const onnx::TensorProto& scalar_b = weights[node->input(1)];
      if (scalar_b.dims_size() != 0 || get_tensor_proto_data_size(scalar_b) != 1) continue;

      float b = get_node_attr_from_input<float>(scalar_b);

      node_reference[node->input(1)] -= 1;

      std::string input = node->input(0);

      node->clear_input();
      node->add_input(input);

      onnx::AttributeProto* attr_with_scalar = node->add_attribute();
      attr_with_scalar->set_name("with_scalar");
      attr_with_scalar->set_i(1);

      onnx::AttributeProto* attr_b = node->add_attribute();
      attr_b->set_name("b");
      attr_b->set_f(b);
    }
  }
}

void fuse_hardswish(onnx::GraphProto* mutable_graph,
                    std::map<std::string, onnx::TensorProto>& weights,
                    std::map<std::string, int>& node_reference, std::set<std::string>& blob_names,
                    int& reduced_node_count) {
  int node_count = mutable_graph->node_size();
  for (int i = 0; i < node_count; i++) {
    onnx::NodeProto* node = mutable_graph->mutable_node(i);

    // HardSwish <= Add(+3) - Clip(0,6) - Mul(X,) - Div(/6)
    // HardSwish <= Add(+3) - Clip(0,6) - Mul(X,) - Mul(*(1/6))
    // HardSwish <= Add(+3) - Clip(0,6) - Mul(X,) - Constant - Div(/6)
    // HardSwish <= Add(+3) - Clip(0,6) - Mul(X,) - Constant - Mul(*(1/6))
    //     out = x * F.relu6(x + 3, inplace=True) / 6
    if (node->op_type() == "Add") {
      if (node_reference[node->output(0)] != 1) continue;

      if (i + 3 >= node_count) continue;

      if (weights.find(node->input(1)) == weights.end()) continue;

      const onnx::TensorProto& add_three = weights[node->input(1)];
      if (add_three.dims_size() != 0 || get_tensor_proto_data_size(add_three) != 1) continue;

      float constant_add_three = get_node_attr_from_input<float>(add_three);
      if (constant_add_three != 3.f) continue;

      onnx::NodeProto* node2 = mutable_graph->mutable_node(i + 1);
      onnx::NodeProto* node3 = mutable_graph->mutable_node(i + 2);
      onnx::NodeProto* node4 = mutable_graph->mutable_node(i + 3);

      if (node4->op_type() == "Constant") {
        if (i + 4 >= node_count) continue;

        node4 = mutable_graph->mutable_node(i + 4);
      }

      if (node2->op_type() != "Clip" || node3->op_type() != "Mul" ||
          (node4->op_type() != "Div" && node4->op_type() != "Mul"))
        continue;

      if (node_reference[node2->output(0)] != 1) continue;

      float relu6_min;
      float relu6_max;
      if (node2->input_size() == 1) {
        relu6_min = get_node_attr_f(*node2, "min", -FLT_MAX);
        relu6_max = get_node_attr_f(*node2, "max", FLT_MAX);
      } else {
        const onnx::TensorProto& min_tp = weights[node2->input(1)];
        const onnx::TensorProto& max_tp = weights[node2->input(2)];

        relu6_min = get_node_attr_from_input<float>(min_tp);
        relu6_max = get_node_attr_from_input<float>(max_tp);
      }
      if (relu6_min != 0.f || relu6_max != 6.f) continue;

      if (node_reference[node3->output(0)] != 1) continue;

      if (node3->input(0) != node->input(0) || node3->input(1) != node2->output(0)) continue;

      if (weights.find(node4->input(1)) == weights.end()) continue;

      const onnx::TensorProto& div_six = weights[node4->input(1)];
      if (div_six.dims_size() != 0 || get_tensor_proto_data_size(div_six) != 1) continue;

      float constant_div_six = get_node_attr_from_input<float>(div_six);
      if (node4->op_type() == "Div" && constant_div_six != 6.f) continue;
      if (node4->op_type() == "Mul" && constant_div_six != 1 / 6.f) continue;

      // reduce
      node->set_op_type("noop_reducedncnn");
      node2->set_op_type("noop_reducedncnn");
      node3->set_op_type("noop_reducedncnn");

      node_reference[node->input(0)] -= 1;
      node_reference[node->input(1)] -= 1;
      node_reference[node->output(0)] -= 1;
      if (node2->input_size() == 3) {
        node_reference[node2->input(1)] -= 1;
        node_reference[node2->input(2)] -= 1;
      }
      node_reference[node2->output(0)] -= 1;
      node_reference[node3->output(0)] -= 1;
      node_reference[node4->input(1)] -= 1;

      blob_names.erase(node->output(0));
      blob_names.erase(node2->output(0));
      blob_names.erase(node3->output(0));

      node4->set_op_type("HardSwish");
      node4->clear_input();
      node4->add_input(node->input(0));

      onnx::AttributeProto* attr_alpha = node4->add_attribute();
      attr_alpha->set_name("alpha");
      attr_alpha->set_f(1.f / 6.f);

      onnx::AttributeProto* attr_beta = node4->add_attribute();
      attr_beta->set_name("beta");
      attr_beta->set_f(3.f / 6.f);

      reduced_node_count += 3;
      i += 3;
    }
  }

  for (int i = 0; i < node_count; i++) {
    onnx::NodeProto* node = mutable_graph->mutable_node(i);

    // HardSwish <= HardSigmoid - Mul
    //     out = x * hsigmoid(x)
    if (node->op_type() == "HardSigmoid") {
      if (node_reference[node->output(0)] != 1) continue;

      float alpha = get_node_attr_f(*node, "alpha", 0.2f);
      float beta = get_node_attr_f(*node, "beta", 0.5f);

      if (i + 1 >= node_count) continue;

      onnx::NodeProto* node2 = mutable_graph->mutable_node(i + 1);

      if (node2->op_type() != "Mul") continue;

      if (node2->input(0) != node->input(0) || node2->input(1) != node->output(0)) continue;

      // reduce
      node->set_op_type("noop_reducedncnn");

      node_reference[node->input(0)] -= 1;
      node_reference[node->output(0)] -= 1;

      blob_names.erase(node->output(0));

      node2->set_op_type("HardSwish");
      node2->clear_input();
      node2->add_input(node->input(0));

      onnx::AttributeProto* attr_alpha = node2->add_attribute();
      attr_alpha->set_name("alpha");
      attr_alpha->set_f(alpha);

      onnx::AttributeProto* attr_beta = node2->add_attribute();
      attr_beta->set_name("beta");
      attr_beta->set_f(beta);

      reduced_node_count += 1;
      i += 1;
    }
  }
}

void fuse_hardsigmoid(onnx::GraphProto* mutable_graph,
                      std::map<std::string, onnx::TensorProto>& weights,
                      std::map<std::string, int>& node_reference, std::set<std::string>& blob_names,
                      int& reduced_node_count) {
  int node_count = mutable_graph->node_size();
  for (int i = 0; i < node_count; i++) {
    onnx::NodeProto* node = mutable_graph->mutable_node(i);

    // HardSigmoid <= Add(+3) - Clip(0,6) - Div(/6)
    // HardSigmoid <= Add(+3) - Clip(0,6) - Mul(*(1/6))
    // HardSigmoid <= Add(+3) - Clip(0,6) - Constant - Div(/6)
    // HardSigmoid <= Add(+3) - Clip(0,6) - Constant - Mul(*(1/6))
    //     out = F.relu6(x + 3, inplace=True) / 6
    if (node->op_type() == "Add") {
      if (node_reference[node->output(0)] != 1) continue;

      if (i + 2 >= node_count) continue;

      if (weights.find(node->input(1)) == weights.end()) continue;

      const onnx::TensorProto& add_three = weights[node->input(1)];
      if (add_three.dims_size() != 0 || get_tensor_proto_data_size(add_three) != 1) continue;

      float constant_add_three = get_node_attr_from_input<float>(add_three);
      if (constant_add_three != 3.f) continue;

      onnx::NodeProto* node2 = mutable_graph->mutable_node(i + 1);
      onnx::NodeProto* node3 = mutable_graph->mutable_node(i + 2);

      if (node3->op_type() == "Constant") {
        if (i + 3 >= node_count) continue;

        node3 = mutable_graph->mutable_node(i + 3);
      }

      if (node2->op_type() != "Clip" || (node3->op_type() != "Div" && node3->op_type() != "Mul"))
        continue;

      if (node_reference[node2->output(0)] != 1) continue;

      float relu6_min;
      float relu6_max;
      if (node2->input_size() == 1) {
        relu6_min = get_node_attr_f(*node2, "min", -FLT_MAX);
        relu6_max = get_node_attr_f(*node2, "max", FLT_MAX);
      } else {
        const onnx::TensorProto& min_tp = weights[node2->input(1)];
        const onnx::TensorProto& max_tp = weights[node2->input(2)];

        relu6_min = get_node_attr_from_input<float>(min_tp);
        relu6_max = get_node_attr_from_input<float>(max_tp);
      }
      if (relu6_min != 0.f || relu6_max != 6.f) continue;

      if (weights.find(node3->input(1)) == weights.end()) continue;

      const onnx::TensorProto& div_six = weights[node3->input(1)];
      if (div_six.dims_size() != 0 || get_tensor_proto_data_size(div_six) != 1) continue;

      float constant_div_six = get_node_attr_from_input<float>(div_six);
      if (node3->op_type() == "Div" && constant_div_six != 6.f) continue;
      if (node3->op_type() == "Mul" && constant_div_six != 1 / 6.f) continue;

      // reduce
      node->set_op_type("noop_reducedncnn");
      node2->set_op_type("noop_reducedncnn");

      node_reference[node->input(1)] -= 1;
      node_reference[node->output(0)] -= 1;
      if (node2->input_size() == 3) {
        node_reference[node2->input(1)] -= 1;
        node_reference[node2->input(2)] -= 1;
      }
      node_reference[node2->output(0)] -= 1;
      node_reference[node3->input(1)] -= 1;

      blob_names.erase(node->output(0));
      blob_names.erase(node2->output(0));

      node3->set_op_type("HardSigmoid");
      node3->clear_input();
      node3->add_input(node->input(0));

      onnx::AttributeProto* attr_alpha = node3->add_attribute();
      attr_alpha->set_name("alpha");
      attr_alpha->set_f(1.f / 6.f);

      onnx::AttributeProto* attr_beta = node3->add_attribute();
      attr_beta->set_name("beta");
      attr_beta->set_f(3.f / 6.f);

      reduced_node_count += 2;
      i += 2;
    }
  }
}

void fuse_swish(onnx::GraphProto* mutable_graph, std::map<std::string, onnx::TensorProto>& weights,
                std::map<std::string, int>& node_reference, std::set<std::string>& blob_names,
                int& reduced_node_count) {
  int node_count = mutable_graph->node_size();
  for (int i = 0; i < node_count; i++) {
    onnx::NodeProto* node = mutable_graph->mutable_node(i);

    // Swish <= Sigmoid - Mul
    //     x * torch.sigmoid(x)
    if (node->op_type() == "Sigmoid") {
      if (node_reference[node->output(0)] != 1) continue;

      if (i + 1 >= node_count) continue;

      onnx::NodeProto* node2 = mutable_graph->mutable_node(i + 1);

      if (node2->op_type() != "Mul") continue;

      if (node2->input(0) != node->input(0) || node2->input(1) != node->output(0)) continue;

      // reduce
      node->set_op_type("noop_reducedncnn");

      node_reference[node->input(0)] -= 1;
      node_reference[node->output(0)] -= 1;

      blob_names.erase(node->output(0));

      node2->set_op_type("Swish");
      node2->clear_input();
      node2->add_input(node->input(0));

      reduced_node_count += 1;
      i += 1;
    }
  }
}

void fuse_batchnorm1d_squeeze_unsqueeze(onnx::GraphProto* mutable_graph,
                                        std::map<std::string, onnx::TensorProto>& weights,
                                        std::map<std::string, int>& node_reference,
                                        std::set<std::string>& blob_names,
                                        int& reduced_node_count) {
  int node_count = mutable_graph->node_size();
  for (int i = 0; i < node_count; i++) {
    onnx::NodeProto* node = mutable_graph->mutable_node(i);

    // BatchNormalization <= Unsqueeze - BatchNormalization - Squeeze
    if (node->op_type() == "Unsqueeze") {
      if (node_reference[node->output(0)] != 1) continue;

      if (i + 2 >= node_count) continue;

      onnx::NodeProto* node2 = mutable_graph->mutable_node(i + 1);
      onnx::NodeProto* node3 = mutable_graph->mutable_node(i + 2);

      if (node2->op_type() != "BatchNormalization" || node3->op_type() != "Squeeze") continue;

      if (node_reference[node2->output(0)] != 1) continue;

      if (node2->input(0) != node->output(0) || node3->input(0) != node2->output(0)) continue;

      // reduce
      node->set_op_type("noop_reducedncnn");
      node3->set_op_type("noop_reducedncnn");

      node_reference[node->output(0)] -= 1;
      node_reference[node2->output(0)] -= 1;

      blob_names.erase(node->output(0));
      blob_names.erase(node2->output(0));

      node2->set_input(0, node->input(0));
      node2->set_output(0, node3->output(0));

      reduced_node_count += 2;
      i += 2;
    }
  }
}

void fuse_unsqueeze_prelu(onnx::GraphProto* mutable_graph,
                          std::map<std::string, onnx::TensorProto>& weights,
                          std::map<std::string, int>& node_reference,
                          std::set<std::string>& blob_names, int& reduced_node_count) {
  int node_count = mutable_graph->node_size();
  for (int i = 0; i < node_count; i++) {
    onnx::NodeProto* node = mutable_graph->mutable_node(i);

    // PReLU <= Unsqueeze - PReLU
    if (node->op_type() == "Unsqueeze") {
      // check weight
      if (weights.find(node->input(0)) == weights.end()) continue;

      onnx::TensorProto& B = weights[node->input(0)];
      if (B.dims_size() != 1) continue;

      if (node_reference[node->output(0)] != 1) continue;

      // axes = (1, 2)
      std::vector<int> axes = get_node_attr_ai(*node, "axes");
      if (axes.size() != 2) continue;
      if (axes[0] != 1 || axes[1] != 2) continue;

      if (i + 1 >= node_count) continue;

      onnx::NodeProto* node2 = mutable_graph->mutable_node(i + 1);

      if (node2->op_type() != "PRelu") continue;

      if (node2->input(1) != node->output(0)) continue;

      // reduce
      node->set_op_type("noop_reducedncnn");

      node_reference[node->output(0)] -= 1;

      blob_names.erase(node->output(0));

      node2->set_input(1, node->input(0));

      reduced_node_count += 1;
      i += 1;
    }
  }
}

void fuse_normalize(onnx::GraphProto* mutable_graph,
                    std::map<std::string, onnx::TensorProto>& weights,
                    std::map<std::string, int>& node_reference, std::set<std::string>& blob_names,
                    int& reduced_node_count) {
  int node_count = mutable_graph->node_size();
  for (int i = 0; i < node_count; i++) {
    onnx::NodeProto* node = mutable_graph->mutable_node(i);

    // Normalize <= X - ReduceL2 - Clip - Expand - Div
    // Normalize <= X - ReduceL2 - Clip - Shape - Expand - Div
    if (node->op_type() == "ReduceL2") {
      if (node_reference[node->output(0)] != 1) continue;

      // axes = (1)
      std::vector<int> axes = get_node_attr_ai(*node, "axes");
      if (axes.size() != 1) continue;
      if (axes[0] != 1) continue;

      if (i + 3 >= node_count) continue;

      onnx::NodeProto* node2 = mutable_graph->mutable_node(i + 1);
      onnx::NodeProto* node3 = mutable_graph->mutable_node(i + 2);
      onnx::NodeProto* node4 = mutable_graph->mutable_node(i + 3);

      bool has_shape_node = node3->op_type() == "Shape";
      onnx::NodeProto* node_shape = 0;
      if (has_shape_node) {
        if (i + 4 >= node_count) continue;

        node_shape = node3;
        node3 = mutable_graph->mutable_node(i + 3);
        node4 = mutable_graph->mutable_node(i + 4);
      }

      if (node2->op_type() != "Clip" || node3->op_type() != "Expand" || node4->op_type() != "Div")
        continue;

      if (node_reference[node2->output(0)] != 1) continue;

      if (node_reference[node3->output(0)] != 1) continue;

      if (node2->input(0) != node->output(0) || node3->input(0) != node2->output(0) ||
          node4->input(0) != node->input(0) || node4->input(1) != node3->output(0))
        continue;

      if (has_shape_node) {
        if (node_shape->input(0) != node->input(0) || node3->input(1) != node_shape->output(0))
          continue;
      }

      // +eps
      float clip_min;
      if (node2->input_size() == 1) {
        clip_min = get_node_attr_f(*node2, "min", -FLT_MAX);
      } else {
        const onnx::TensorProto& min_tp = weights[node2->input(1)];

        clip_min = get_node_attr_from_input<float>(min_tp);
      }

      // reduce
      node->set_op_type("noop_reducedncnn");
      node2->set_op_type("noop_reducedncnn");
      if (has_shape_node) {
        node_shape->set_op_type("noop_reducedncnn");
      }
      node3->set_op_type("noop_reducedncnn");

      node_reference[node->input(0)] -= has_shape_node ? 2 : 1;
      node_reference[node->output(0)] -= 1;
      node_reference[node2->output(0)] -= 1;
      if (has_shape_node) {
        node_reference[node_shape->output(0)] -= 1;
      }
      node_reference[node3->output(0)] -= 1;
      if (node3->input_size() == 2) {
        node_reference[node3->input(1)] -= 1;
      }

      blob_names.erase(node->output(0));
      blob_names.erase(node2->output(0));
      if (has_shape_node) {
        blob_names.erase(node_shape->output(0));
      }
      blob_names.erase(node3->output(0));

      node4->set_op_type("Normalize");
      node4->clear_input();
      node4->add_input(node->input(0));

      onnx::AttributeProto* attr_alpha = node4->add_attribute();
      attr_alpha->set_name("eps");
      attr_alpha->set_f(clip_min);

      reduced_node_count += has_shape_node ? 4 : 3;
      i += has_shape_node ? 4 : 3;
    }
  }
}

void fuse_groupnorm(onnx::GraphProto* mutable_graph,
                    std::map<std::string, onnx::TensorProto>& weights,
                    std::map<std::string, int>& node_reference, std::set<std::string>& blob_names,
                    int& reduced_node_count) {
  int node_count = mutable_graph->node_size();
  for (int i = 0; i < node_count; i++) {
    onnx::NodeProto* node = mutable_graph->mutable_node(i);

    // GroupNorm <= X - Reshape - InstanceNormalization - Reshape - Mul - Add
    if (node->op_type() == "Reshape") {
      if (node_reference[node->output(0)] != 1) continue;

      std::vector<int> shape;
      if (node->input_size() == 1) {
        shape = get_node_attr_ai(*node, "shape");
      } else {
        // skip weight reshape
        if (weights.find(node->input(1)) == weights.end()) continue;

        shape = get_node_attr_from_input_ai(weights[node->input(1)]);
      }

      // 0, group, -1
      if (shape.size() != 3) continue;

      if (shape[0] != 0 || shape[2] != -1) continue;

      int groups = shape[1];

      if (i + 4 >= node_count) continue;

      onnx::NodeProto* node2 = mutable_graph->mutable_node(i + 1);
      onnx::NodeProto* node3 = mutable_graph->mutable_node(i + 2);
      onnx::NodeProto* node4 = mutable_graph->mutable_node(i + 3);
      onnx::NodeProto* node5 = mutable_graph->mutable_node(i + 4);

      if (node2->op_type() != "InstanceNormalization" || node3->op_type() != "Reshape" ||
          node4->op_type() != "Mul" || node5->op_type() != "Add")
        continue;

      if (node_reference[node2->output(0)] != 1) continue;

      if (node_reference[node3->output(0)] != 1) continue;

      if (node_reference[node4->output(0)] != 1) continue;

      if (node2->input(0) != node->output(0) || node3->input(0) != node2->output(0) ||
          node4->input(0) != node3->output(0) || node5->input(0) != node4->output(0))
        continue;

      // +eps
      float eps = get_node_attr_f(*node2, "epsilon", 1e-05f);

      // InstanceNormalization S=1 B=0
      std::vector<float> S = get_node_attr_from_input_af(weights[node2->input(1)]);
      std::vector<float> B = get_node_attr_from_input_af(weights[node2->input(2)]);
      if ((int)S.size() != groups || (int)B.size() != groups) continue;

      bool instancenorm_affine = false;
      for (int j = 0; j < groups; j++) {
        if (S[j] != 1.f || B[j] != 0.f) {
          instancenorm_affine = true;
          break;
        }
      }

      if (instancenorm_affine) continue;

      std::vector<int> shape2;
      if (node3->input_size() == 1) {
        shape2 = get_node_attr_ai(*node3, "shape");
      } else {
        // skip weight reshape
        if (weights.find(node3->input(1)) == weights.end()) continue;

        shape2 = get_node_attr_from_input_ai(weights[node3->input(1)]);
      }

      // 1, channels, w, h
      if (shape2.size() != 4) continue;

      if (shape2[0] != 1) continue;

      int channels = shape2[1];

      // affine
      std::vector<float> affine_S = get_node_attr_from_input_af(weights[node4->input(1)]);
      std::vector<float> affine_B = get_node_attr_from_input_af(weights[node5->input(1)]);
      if (affine_S.size() == 1 && affine_S[0] == 1.f && affine_B.size() == 1 &&
          affine_B[0] == 0.f) {
        // no affine
      } else if ((int)affine_S.size() != channels && (int)affine_B.size() != channels) {
        // we only allow per-channel affine
        continue;
      }

      // reduce
      node->set_op_type("noop_reducedncnn");
      node2->set_op_type("noop_reducedncnn");
      node3->set_op_type("noop_reducedncnn");
      node4->set_op_type("noop_reducedncnn");

      if (node->input_size() == 2) {
        node_reference[node->input(1)] -= 1;
      }
      node_reference[node->output(0)] -= 1;
      node_reference[node2->input(1)] -= 1;
      node_reference[node2->input(2)] -= 1;
      node_reference[node2->output(0)] -= 1;
      if (node3->input_size() == 2) {
        node_reference[node3->input(1)] -= 1;
      }
      node_reference[node3->output(0)] -= 1;
      node_reference[node4->output(0)] -= 1;

      blob_names.erase(node->output(0));
      blob_names.erase(node2->output(0));
      blob_names.erase(node3->output(0));
      blob_names.erase(node4->output(0));

      std::string affine_scale = node4->input(1);
      std::string affine_bias = node5->input(1);

      node5->set_op_type("GroupNorm");
      node5->clear_input();
      node5->add_input(node->input(0));
      node5->add_input(affine_scale);
      node5->add_input(affine_bias);

      onnx::AttributeProto* attr_groups = node5->add_attribute();
      attr_groups->set_name("groups");
      attr_groups->set_i(groups);

      onnx::AttributeProto* attr_channels = node5->add_attribute();
      attr_channels->set_name("channels");
      attr_channels->set_i(channels);

      onnx::AttributeProto* attr_eps = node5->add_attribute();
      attr_eps->set_name("epsilon");
      attr_eps->set_f(eps);

      onnx::AttributeProto* attr_affine = node5->add_attribute();
      attr_affine->set_name("affine");
      attr_affine->set_i(1);

      reduced_node_count += 4;
      i += 4;
    }
  }
}

void fuse_layernorm(onnx::GraphProto* mutable_graph,
                    std::map<std::string, onnx::TensorProto>& weights,
                    std::map<std::string, int>& node_reference, std::set<std::string>& blob_names,
                    int& reduced_node_count) {
  int node_count = mutable_graph->node_size();
  for (int i = 0; i < node_count; i++) {
    onnx::NodeProto* node = mutable_graph->mutable_node(i);

    // LayerNorm <= X - ReduceMean - Sub - Pow - ReduceMean - Add - Sqrt - Div
    // LayerNorm <= X - ReduceMean - Sub - Pow - ReduceMean - Add - Sqrt - Div -
    // Mul - Add
    if (node->op_type() == "ReduceMean") {
      if (node_reference[node->output(0)] != 1) continue;

      std::vector<int> axes = get_node_attr_ai(*node, "axes");

      // -1
      // -2 -1
      if (axes.size() != 1 && axes.size() != 2) continue;

      int normed_axes = (int)axes.size();
      if (normed_axes == 1 && axes[0] != -1) continue;
      if (normed_axes == 2 && (axes[0] != -2 || axes[1] != -1)) continue;

      if (i + 6 >= node_count) continue;

      onnx::NodeProto* node2 = mutable_graph->mutable_node(i + 1);
      onnx::NodeProto* node3 = mutable_graph->mutable_node(i + 2);
      onnx::NodeProto* node4 = mutable_graph->mutable_node(i + 3);
      onnx::NodeProto* node5 = mutable_graph->mutable_node(i + 4);
      onnx::NodeProto* node6 = mutable_graph->mutable_node(i + 5);
      onnx::NodeProto* node7 = mutable_graph->mutable_node(i + 6);

      if (node2->op_type() != "Sub" || node3->op_type() != "Pow" ||
          node4->op_type() != "ReduceMean" || node5->op_type() != "Add" ||
          node6->op_type() != "Sqrt" || node7->op_type() != "Div")
        continue;

      if (node_reference[node2->output(0)] != 2) continue;

      if (node_reference[node3->output(0)] != 1) continue;

      if (node_reference[node4->output(0)] != 1) continue;

      if (node_reference[node5->output(0)] != 1) continue;

      if (node_reference[node6->output(0)] != 1) continue;

      if (node2->input(0) != node->input(0) || node2->input(1) != node->output(0) ||
          node3->input(0) != node2->output(0) || node4->input(0) != node3->output(0) ||
          node5->input(0) != node4->output(0) || node6->input(0) != node5->output(0) ||
          node7->input(0) != node2->output(0) || node7->input(1) != node6->output(0))
        continue;

      if (weights.find(node3->input(1)) == weights.end()) continue;

      const onnx::TensorProto& pow_two = weights[node3->input(1)];
      if (pow_two.dims_size() != 0 || get_tensor_proto_data_size(pow_two) != 1) continue;

      float constant_pow_two = get_node_attr_from_input<float>(pow_two);
      if (constant_pow_two != 2.f) continue;

      std::vector<int> axes4 = get_node_attr_ai(*node4, "axes");

      // -1
      // -2 -1
      if ((int)axes4.size() != normed_axes) continue;

      if (normed_axes == 1 && axes4[0] != -1) continue;
      if (normed_axes == 2 && (axes4[0] != -2 || axes4[1] != -1)) continue;

      if (weights.find(node5->input(1)) == weights.end()) continue;

      const onnx::TensorProto& add_eps = weights[node5->input(1)];
      if (add_eps.dims_size() != 0 || get_tensor_proto_data_size(add_eps) != 1) continue;

      float eps = get_node_attr_from_input<float>(add_eps);

      int affine = 0;
      while (i + 8 < node_count) {
        onnx::NodeProto* node8 = mutable_graph->mutable_node(i + 7);
        onnx::NodeProto* node9 = mutable_graph->mutable_node(i + 8);

        if (node8->op_type() != "Mul" || node9->op_type() != "Add") break;

        if (node_reference[node7->output(0)] != 1) break;

        if (node_reference[node8->output(0)] != 1) break;

        if (node8->input(0) != node7->output(0) || node9->input(0) != node8->output(0)) break;

        // affine
        std::vector<float> affine_S = get_node_attr_from_input_af(weights[node8->input(1)]);
        std::vector<float> affine_B = get_node_attr_from_input_af(weights[node9->input(1)]);
        if (affine_S.size() != affine_B.size()) break;

        affine = 1;
        break;
      }

      // reduce
      node->set_op_type("noop_reducedncnn");
      node2->set_op_type("noop_reducedncnn");
      node3->set_op_type("noop_reducedncnn");
      node4->set_op_type("noop_reducedncnn");
      node5->set_op_type("noop_reducedncnn");
      node6->set_op_type("noop_reducedncnn");

      node_reference[node->input(0)] -= 1;
      node_reference[node2->input(0)] -= 1;
      node_reference[node2->input(1)] -= 1;
      node_reference[node3->input(0)] -= 1;
      node_reference[node3->input(1)] -= 1;
      node_reference[node4->input(0)] -= 1;
      node_reference[node5->input(0)] -= 1;
      node_reference[node5->input(1)] -= 1;
      node_reference[node6->input(0)] -= 1;
      node_reference[node7->input(0)] -= 1;
      node_reference[node7->input(1)] -= 1;

      blob_names.erase(node->output(0));
      blob_names.erase(node2->output(0));
      blob_names.erase(node3->output(0));
      blob_names.erase(node4->output(0));
      blob_names.erase(node5->output(0));
      blob_names.erase(node6->output(0));

      node_reference[node->input(0)] += 1;

      if (affine == 0) {
        node7->set_op_type("LayerNorm");
        node7->clear_input();
        node7->add_input(node->input(0));

        onnx::AttributeProto* attr_eps = node7->add_attribute();
        attr_eps->set_name("epsilon");
        attr_eps->set_f(eps);

        onnx::AttributeProto* attr_affine = node7->add_attribute();
        attr_affine->set_name("affine");
        attr_affine->set_i(affine);

        reduced_node_count += 6;
        i += 6;
      } else  // if (affine == 1)
      {
        onnx::NodeProto* node8 = mutable_graph->mutable_node(i + 7);
        onnx::NodeProto* node9 = mutable_graph->mutable_node(i + 8);

        node7->set_op_type("noop_reducedncnn");
        node8->set_op_type("noop_reducedncnn");

        node_reference[node8->input(0)] -= 1;
        node_reference[node9->input(0)] -= 1;

        blob_names.erase(node7->output(0));
        blob_names.erase(node8->output(0));

        std::string affine_scale = node8->input(1);
        std::string affine_bias = node9->input(1);

        node9->set_op_type("LayerNorm");
        node9->clear_input();
        node9->add_input(node->input(0));
        node9->add_input(affine_scale);
        node9->add_input(affine_bias);

        onnx::AttributeProto* attr_eps = node9->add_attribute();
        attr_eps->set_name("epsilon");
        attr_eps->set_f(eps);

        onnx::AttributeProto* attr_affine = node9->add_attribute();
        attr_affine->set_name("affine");
        attr_affine->set_i(affine);

        reduced_node_count += 8;
        i += 8;
      }
    }
  }
}

void fuse_flatten(onnx::GraphProto* mutable_graph,
                  std::map<std::string, onnx::TensorProto>& weights,
                  std::map<std::string, int>& node_reference, std::set<std::string>& blob_names,
                  int& reduced_node_count) {
  int node_count = mutable_graph->node_size();
  for (int i = 0; i < node_count; i++) {
    onnx::NodeProto* node = mutable_graph->mutable_node(i);

    // Flatten <= X - Shape - Gather - Constant - Unsqueeze - Unsqueeze - Concat
    // - Reshape
    if (node->op_type() == "Shape") {
      if (node_reference[node->output(0)] != 1) continue;

      if (i + 6 >= node_count) continue;

      onnx::NodeProto* node2 = mutable_graph->mutable_node(i + 1);
      onnx::NodeProto* node3 = mutable_graph->mutable_node(i + 2);
      onnx::NodeProto* node4 = mutable_graph->mutable_node(i + 3);
      onnx::NodeProto* node5 = mutable_graph->mutable_node(i + 4);
      onnx::NodeProto* node6 = mutable_graph->mutable_node(i + 5);
      onnx::NodeProto* node7 = mutable_graph->mutable_node(i + 6);

      if (node2->op_type() != "Gather" || node3->op_type() != "Constant" ||
          node4->op_type() != "Unsqueeze" || node5->op_type() != "Unsqueeze" ||
          node6->op_type() != "Concat" || node7->op_type() != "Reshape")
        continue;

      if (node_reference[node2->output(0)] != 1) continue;

      //             if (node_reference[node3->output(0)] != 1)
      //                 continue;

      if (node_reference[node4->output(0)] != 1) continue;

      if (node_reference[node5->output(0)] != 1) continue;

      if (node_reference[node6->output(0)] != 1) continue;

      if (node2->input(0) != node->output(0) || node4->input(0) != node2->output(0) ||
          node5->input(0) != node3->output(0) || node6->input(0) != node4->output(0) ||
          node6->input(1) != node5->output(0) || node7->input(0) != node->input(0) ||
          node7->input(1) != node6->output(0))
        continue;

      // axis = 0
      int gather_axis = get_node_attr_i(*node2, "axis");
      if (gather_axis != 0) continue;

      // indices = 0
      if (weights.find(node2->input(1)) == weights.end()) continue;

      std::vector<int> gather_indices = get_node_attr_from_input_ai(weights[node2->input(1)]);
      if (gather_indices.size() != 1 || gather_indices[0] != 0) continue;

      // axes = (0)
      std::vector<int> unsqueeze_axes = get_node_attr_ai(*node4, "axes");
      if (unsqueeze_axes.size() != 1) continue;
      if (unsqueeze_axes[0] != 0) continue;

      // axes = (0)
      std::vector<int> unsqueeze2_axes = get_node_attr_ai(*node5, "axes");
      if (unsqueeze2_axes.size() != 1) continue;
      if (unsqueeze2_axes[0] != 0) continue;

      // data = -1
      if (weights.find(node5->input(0)) == weights.end()) continue;

      std::vector<int> unsqueeze2_data = get_node_attr_from_input_ai(weights[node5->input(0)]);
      if (unsqueeze2_data.size() != 1 || unsqueeze2_data[0] != -1) continue;

      // axis = 0
      int concat_axis = get_node_attr_i(*node6, "axis");
      if (concat_axis != 0) continue;

      // reduce
      node->set_op_type("noop_reducedncnn");
      node2->set_op_type("noop_reducedncnn");
      //             node3->set_op_type("noop_reducedncnn");
      node4->set_op_type("noop_reducedncnn");
      node5->set_op_type("noop_reducedncnn");
      node6->set_op_type("noop_reducedncnn");

      node_reference[node->input(0)] -= 1;
      node_reference[node->output(0)] -= 1;
      node_reference[node2->input(1)] -= 1;
      node_reference[node2->output(0)] -= 1;
      //             node_reference[node3->output(0)] -= 1;
      node_reference[node4->output(0)] -= 1;
      node_reference[node5->input(0)] -= 1;
      node_reference[node5->output(0)] -= 1;
      node_reference[node6->output(0)] -= 1;

      blob_names.erase(node->output(0));
      blob_names.erase(node2->output(0));
      //             blob_names.erase(node3->output(0));
      blob_names.erase(node4->output(0));
      blob_names.erase(node5->output(0));
      blob_names.erase(node6->output(0));

      node7->set_op_type("Flatten");
      node7->clear_input();
      node7->add_input(node->input(0));

      reduced_node_count += 5;
      i += 5;
    }
  }
}

void fuse_pixelshuffle(onnx::GraphProto* mutable_graph,
                       std::map<std::string, onnx::TensorProto>& weights,
                       std::map<std::string, int>& node_reference,
                       std::set<std::string>& blob_names, int& reduced_node_count) {
  int node_count = mutable_graph->node_size();
  for (int i = 0; i < node_count; i++) {
    onnx::NodeProto* node = mutable_graph->mutable_node(i);

    // PixelShuffle <= Reshape - Transpose - Reshape
    // PixelShuffle <= Reshape - Transpose - Constant - Reshape
    if (node->op_type() == "Reshape") {
      if (node_reference[node->output(0)] != 1) continue;

      std::vector<int> shape;
      if (node->input_size() == 1) {
        shape = get_node_attr_ai(*node, "shape");
      } else {
        // skip weight reshape
        if (weights.find(node->input(1)) == weights.end()) continue;

        shape = get_node_attr_from_input_ai(weights[node->input(1)]);
      }

      // -1, 3, upscale_factor, upscale_factor, height, width
      if (shape.size() != 6) continue;

      if (shape[0] != 1 && shape[0] != -1) continue;

      if (shape[2] != shape[3]) continue;

      if (i + 2 >= node_count) continue;

      onnx::NodeProto* node2 = mutable_graph->mutable_node(i + 1);
      onnx::NodeProto* node3 = mutable_graph->mutable_node(i + 2);

      if (node3->op_type() == "Constant") {
        if (i + 3 >= node_count) continue;

        node3 = mutable_graph->mutable_node(i + 3);
      }

      if (node2->op_type() != "Transpose" || node3->op_type() != "Reshape") continue;

      if (node_reference[node2->output(0)] != 1) continue;

      // 0 1 4 2 5 3
      std::vector<int> perm = get_node_attr_ai(*node2, "perm");
      if (perm.size() != 6) continue;

      if (perm[0] != 0 || perm[1] != 1 || perm[2] != 4 || perm[3] != 2 || perm[4] != 5 ||
          perm[5] != 3)
        continue;

      std::vector<int> shape3;
      if (node3->input_size() == 1) {
        shape3 = get_node_attr_ai(*node3, "shape");
      } else {
        // skip weight reshape
        if (weights.find(node3->input(1)) == weights.end()) continue;

        shape3 = get_node_attr_from_input_ai(weights[node3->input(1)]);
      }

      // -1, 3, height, width
      if (shape3.size() != 4) continue;

      if (shape3[0] != 1 && shape3[0] != -1) continue;

      if (shape3[1] != shape[1] || shape3[2] != shape[2] * shape[4] ||
          shape3[3] != shape[3] * shape[5])
        continue;

      // reduce
      node->set_op_type("noop_reducedncnn");
      node2->set_op_type("noop_reducedncnn");

      if (node->input_size() == 2) {
        node_reference[node->input(1)] -= 1;
      }
      node_reference[node->output(0)] -= 1;
      node_reference[node2->output(0)] -= 1;
      if (node3->input_size() == 2) {
        node_reference[node3->input(1)] -= 1;
      }

      blob_names.erase(node->output(0));
      blob_names.erase(node2->output(0));

      node3->set_op_type("PixelShuffle");
      node3->set_input(0, node->input(0));

      onnx::AttributeProto* attr_group = node3->add_attribute();
      attr_group->set_name("scale_factor");
      attr_group->set_i(shape[2]);

      reduced_node_count += 2;
      i += 2;
    }
  }
}

void fuse_reorg(onnx::GraphProto* mutable_graph, std::map<std::string, onnx::TensorProto>& weights,
                std::map<std::string, int>& node_reference, std::set<std::string>& blob_names,
                int& reduced_node_count) {
  int node_count = mutable_graph->node_size();
  for (int i = 0; i < node_count; i++) {
    onnx::NodeProto* node = mutable_graph->mutable_node(i);

    // PixelShuffle <= Reshape - Transpose - Reshape
    // PixelShuffle <= Reshape - Transpose - Constant - Reshape
    if (node->op_type() == "Reshape") {
      if (node_reference[node->output(0)] != 1) continue;

      std::vector<int> shape;
      if (node->input_size() == 1) {
        shape = get_node_attr_ai(*node, "shape");
      } else {
        // skip weight reshape
        if (weights.find(node->input(1)) == weights.end()) continue;

        shape = get_node_attr_from_input_ai(weights[node->input(1)]);
      }

      // -1, 3, out_height, block_size, out_width, block_size
      if (shape.size() != 6) continue;

      if (shape[0] != 1 && shape[0] != -1) continue;

      if (shape[3] != shape[5]) continue;

      if (i + 2 >= node_count) continue;

      onnx::NodeProto* node2 = mutable_graph->mutable_node(i + 1);
      onnx::NodeProto* node3 = mutable_graph->mutable_node(i + 2);

      if (node3->op_type() == "Constant") {
        if (i + 3 >= node_count) continue;

        node3 = mutable_graph->mutable_node(i + 3);
      }

      if (node2->op_type() != "Transpose" || node3->op_type() != "Reshape") continue;

      if (node_reference[node2->output(0)] != 1) continue;

      // 0 1 3 5 2 4
      std::vector<int> perm = get_node_attr_ai(*node2, "perm");
      if (perm.size() != 6) continue;

      if (perm[0] != 0 || perm[1] != 1 || perm[2] != 3 || perm[3] != 5 || perm[4] != 2 ||
          perm[5] != 4)
        continue;

      std::vector<int> shape3;
      if (node3->input_size() == 1) {
        shape3 = get_node_attr_ai(*node3, "shape");
      } else {
        // skip weight reshape
        if (weights.find(node3->input(1)) == weights.end()) continue;

        shape3 = get_node_attr_from_input_ai(weights[node3->input(1)]);
      }

      // -1, out_channels, out_height, out_width
      if (shape3.size() != 4) continue;

      if (shape3[0] != 1 && shape3[0] != -1) continue;

      if (shape3[1] != shape[1] * shape[3] * shape[5] || shape3[2] != shape[2] ||
          shape3[3] != shape[4])
        continue;

      // reduce
      node->set_op_type("noop_reducedncnn");
      node2->set_op_type("noop_reducedncnn");

      if (node->input_size() == 2) {
        node_reference[node->input(1)] -= 1;
      }
      node_reference[node->output(0)] -= 1;
      node_reference[node2->output(0)] -= 1;
      if (node3->input_size() == 2) {
        node_reference[node3->input(1)] -= 1;
      }

      blob_names.erase(node->output(0));
      blob_names.erase(node2->output(0));

      node3->set_op_type("Reorg");
      node3->set_input(0, node->input(0));

      onnx::AttributeProto* attr_group = node3->add_attribute();
      attr_group->set_name("stride");
      attr_group->set_i(shape[3]);

      reduced_node_count += 2;
      i += 2;
    }
  }
}

void fuse_expand_broadcast(onnx::GraphProto* mutable_graph,
                           std::map<std::string, onnx::TensorProto>& weights,
                           std::map<std::string, int>& node_reference,
                           std::set<std::string>& blob_names, int& reduced_node_count) {
  int node_count = mutable_graph->node_size();
  for (int i = 0; i < node_count; i++) {
    onnx::NodeProto* node = mutable_graph->mutable_node(i);

    // Add/Sub/Mul/Div/Min/Max <= Expand - Add/Sub/Mul/Div/Min/Max
    if (node->op_type() == "Expand") {
      if (node_reference[node->output(0)] != 1) continue;

      if (i + 1 >= node_count) continue;

      onnx::NodeProto* node2 = mutable_graph->mutable_node(i + 1);

      if (node2->op_type() != "Add" && node2->op_type() != "Sub" && node2->op_type() != "Mul" &&
          node2->op_type() != "Div" && node2->op_type() != "Min" && node2->op_type() != "Max")
        continue;

      if (node2->input(1) != node->output(0) && node2->input(0) != node->output(0)) continue;

      // reduce
      node->set_op_type("noop_reducedncnn");

      node_reference[node->output(0)] -= 1;
      if (node->input_size() == 2) {
        node_reference[node->input(1)] -= 1;
      }

      blob_names.erase(node->output(0));

      if (node2->input(0) == node->output(0)) {
        node2->set_input(0, node->input(0));
      } else {
        node2->set_input(1, node->input(0));
      }

      reduced_node_count += 1;
      i += 1;
    }
  }
}

void fuse_lstm_gru_rnn(onnx::GraphProto* mutable_graph,
                       std::map<std::string, onnx::TensorProto>& weights,
                       std::map<std::string, int>& node_reference,
                       std::set<std::string>& blob_names, int& reduced_node_count) {
  int node_count = mutable_graph->node_size();
  for (int i = 0; i < node_count; i++) {
    onnx::NodeProto* node = mutable_graph->mutable_node(i);

    // LSTM(bi) <= LSTM(bi) - Transpose - Reshape - Transpose
    // or LSTM(bi) <= LSTM(bi) - Transpose Constant - Reshape - Transpose
    if (node->op_type() == "LSTM" || node->op_type() == "GRU" || node->op_type() == "RNN") {
      if (node_reference[node->output(0)] != 1) continue;

      if (i + 2 >= node_count) continue;

      onnx::NodeProto* node2 = mutable_graph->mutable_node(i + 1);
      onnx::NodeProto* node3 = mutable_graph->mutable_node(i + 2);

      // skip if second ops is constant
      if (node3->op_type() == "Constant") {
        if (i + 3 >= node_count) continue;
        node3 = mutable_graph->mutable_node(i + 3);
        i += 1;
      }

      if (node2->op_type() != "Transpose" || node3->op_type() != "Reshape") continue;

      if (node_reference[node2->output(0)] != 1) continue;

      if (node2->input(0) != node->output(0) || node3->input(0) != node2->output(0)) continue;

      std::string direction = get_node_attr_s(*node, "direction");
      if (direction != "bidirectional") continue;

      // 0 2 1 3
      std::vector<int> perm = get_node_attr_ai(*node2, "perm");
      if (perm.size() != 4) continue;

      if (perm[0] != 0 || perm[1] != 2 || perm[2] != 1 || perm[3] != 3) continue;

      std::vector<int> shape;
      if (node3->input_size() == 1) {
        shape = get_node_attr_ai(*node3, "shape");
      } else {
        // skip weight reshape
        if (weights.find(node3->input(1)) == weights.end()) continue;

        shape = get_node_attr_from_input_ai(weights[node3->input(1)]);
      }

      // 0 0 -1
      if (shape.size() != 3) continue;

      if (shape[0] != 0 || shape[1] != 0 || shape[2] != -1) continue;

      // reduce
      node2->set_op_type("noop_reducedncnn");
      node3->set_op_type("noop_reducedncnn");

      node_reference[node->output(0)] -= 1;
      node_reference[node2->output(0)] -= 1;
      if (node3->input_size() == 2) {
        node_reference[node3->input(1)] -= 1;
      }

      blob_names.erase(node->output(0));
      blob_names.erase(node2->output(0));

      node->set_output(0, node3->output(0));

      reduced_node_count += 2;
      i += 2;

      if (i + 1 < node_count) {
        if (node_reference[node3->output(0)] != 1) continue;

        onnx::NodeProto* node4 = mutable_graph->mutable_node(i + 1);

        if (node4->op_type() != "Transpose") continue;

        if (node4->input(0) != node->output(0)) continue;

        // 1 0 2
        std::vector<int> perm4 = get_node_attr_ai(*node4, "perm");
        if (perm4.size() != 3) continue;

        if (perm4[0] != 1 || perm4[1] != 0 || perm4[2] != 2) continue;

        // reduce
        node4->set_op_type("noop_reducedncnn");

        node_reference[node->output(0)] -= 1;

        blob_names.erase(node->output(0));

        node->set_output(0, node4->output(0));

        reduced_node_count += 1;
        i += 1;
      }
    }
  }

  for (int i = 0; i < node_count; i++) {
    onnx::NodeProto* node = mutable_graph->mutable_node(i);

    // LSTM(uni) <= LSTM(uni) - Squeeze - Transpose
    if (node->op_type() == "LSTM" || node->op_type() == "GRU" || node->op_type() == "RNN") {
      if (node_reference[node->output(0)] != 1) continue;

      if (i + 1 >= node_count) continue;

      onnx::NodeProto* node2 = mutable_graph->mutable_node(i + 1);

      if (node2->op_type() != "Squeeze") continue;

      if (node2->input(0) != node->output(0)) continue;

      std::string direction = get_node_attr_s(*node, "direction");
      if (direction == "bidirectional") continue;

      // 1
      std::vector<int> axes = get_node_attr_ai(*node2, "axes");
      if (axes.size() != 1) continue;

      if (axes[0] != 1) continue;

      // reduce
      node2->set_op_type("noop_reducedncnn");

      node_reference[node->output(0)] -= 1;

      blob_names.erase(node->output(0));

      node->set_output(0, node2->output(0));

      reduced_node_count += 1;
      i += 1;

      if (i + 1 < node_count) {
        if (node_reference[node2->output(0)] != 1) continue;

        onnx::NodeProto* node3 = mutable_graph->mutable_node(i + 1);

        if (node3->op_type() != "Transpose") continue;

        if (node3->input(0) != node->output(0)) continue;

        // 1 0 2
        std::vector<int> perm4 = get_node_attr_ai(*node3, "perm");
        if (perm4.size() != 3) continue;

        if (perm4[0] != 1 || perm4[1] != 0 || perm4[2] != 2) continue;

        // reduce
        node3->set_op_type("noop_reducedncnn");

        node_reference[node->output(0)] -= 1;

        blob_names.erase(node->output(0));

        node->set_output(0, node3->output(0));

        reduced_node_count += 1;
        i += 1;
      }
    }
  }

  for (int i = 0; i < node_count; i++) {
    onnx::NodeProto* node = mutable_graph->mutable_node(i);

    // LSTM <= Transpose - LSTM
    if (node->op_type() == "Transpose") {
      if (node_reference[node->output(0)] != 1) continue;

      // 1 0 2
      std::vector<int> perm = get_node_attr_ai(*node, "perm");
      if (perm.size() != 3) continue;

      if (perm[0] != 1 || perm[1] != 0 || perm[2] != 2) continue;

      if (i + 1 >= node_count) continue;

      onnx::NodeProto* node2 = mutable_graph->mutable_node(i + 1);

      if (node2->op_type() != "LSTM" && node->op_type() != "GRU" && node->op_type() != "RNN")
        continue;

      if (node2->input(0) != node->output(0)) continue;

      // reduce
      node->set_op_type("noop_reducedncnn");

      node_reference[node->output(0)] -= 1;

      blob_names.erase(node->output(0));

      node2->set_input(0, node->input(0));

      reduced_node_count += 1;
      i += 1;
    }
  }
}

void fuse_multiheadattention(onnx::GraphProto* mutable_graph,
                             std::map<std::string, onnx::TensorProto>& weights,
                             std::map<std::string, int>& node_reference,
                             std::set<std::string>& blob_names, int& reduced_node_count) {
  int node_count = mutable_graph->node_size();
  for (int i = 0; i < node_count; i++) {
    onnx::NodeProto* node = mutable_graph->mutable_node(i);

    // MultiHeadAttention <= MatMul(q) - Add
    //                      - MatMul(k) - Add
    //                      - MatMul(v) - Add
    //                      - Mul
    //                      - Reshape - Transpose
    //                      - Reshape - Reshape - Transpose - Transpose
    //                      - Gemm - Softmax - Gemm - Transpose - Reshape -
    //                      MatMul - Add
    if (node->op_type() == "MatMul") {
      if (i + 19 >= node_count) continue;

      if (node_reference[node->output(0)] != 1) continue;

      onnx::NodeProto* node2 = mutable_graph->mutable_node(i + 1);
      onnx::NodeProto* node3 = mutable_graph->mutable_node(i + 2);
      onnx::NodeProto* node4 = mutable_graph->mutable_node(i + 3);
      onnx::NodeProto* node5 = mutable_graph->mutable_node(i + 4);
      onnx::NodeProto* node6 = mutable_graph->mutable_node(i + 5);
      onnx::NodeProto* node7 = mutable_graph->mutable_node(i + 6);
      onnx::NodeProto* node8 = mutable_graph->mutable_node(i + 7);
      onnx::NodeProto* node9 = mutable_graph->mutable_node(i + 8);
      onnx::NodeProto* node10 = mutable_graph->mutable_node(i + 9);
      onnx::NodeProto* node11 = mutable_graph->mutable_node(i + 10);
      onnx::NodeProto* node12 = mutable_graph->mutable_node(i + 11);
      onnx::NodeProto* node13 = mutable_graph->mutable_node(i + 12);
      onnx::NodeProto* node14 = mutable_graph->mutable_node(i + 13);
      onnx::NodeProto* node15 = mutable_graph->mutable_node(i + 14);
      onnx::NodeProto* node16 = mutable_graph->mutable_node(i + 15);
      onnx::NodeProto* node17 = mutable_graph->mutable_node(i + 16);
      onnx::NodeProto* node18 = mutable_graph->mutable_node(i + 17);
      onnx::NodeProto* node19 = mutable_graph->mutable_node(i + 18);
      onnx::NodeProto* node20 = mutable_graph->mutable_node(i + 19);

      if (node2->op_type() != "Add" || node3->op_type() != "MatMul" || node4->op_type() != "Add" ||
          node5->op_type() != "MatMul" || node6->op_type() != "Add" || node7->op_type() != "Mul" ||
          node8->op_type() != "Reshape" || node9->op_type() != "Transpose" ||
          node10->op_type() != "Reshape" || node11->op_type() != "Reshape" ||
          node12->op_type() != "Transpose" || node13->op_type() != "Transpose" ||
          node14->op_type() != "MatMul" || node15->op_type() != "Softmax" ||
          node16->op_type() != "MatMul" || node17->op_type() != "Transpose" ||
          node18->op_type() != "Reshape" || node19->op_type() != "MatMul" ||
          node20->op_type() != "Add")
        continue;

      if (node_reference[node2->output(0)] != 1 || node_reference[node3->output(0)] != 1 ||
          node_reference[node4->output(0)] != 1 || node_reference[node5->output(0)] != 1 ||
          node_reference[node6->output(0)] != 1 || node_reference[node7->output(0)] != 1 ||
          node_reference[node8->output(0)] != 1 || node_reference[node9->output(0)] != 1 ||
          node_reference[node10->output(0)] != 1 || node_reference[node11->output(0)] != 1 ||
          node_reference[node12->output(0)] != 1 || node_reference[node13->output(0)] != 1 ||
          node_reference[node14->output(0)] != 1 || node_reference[node15->output(0)] != 1 ||
          node_reference[node16->output(0)] != 1 || node_reference[node17->output(0)] != 1 ||
          node_reference[node18->output(0)] != 1 || node_reference[node19->output(0)] != 1)
        continue;

      if (node2->input(0) != node->output(0) || node4->input(0) != node3->output(0) ||
          node6->input(0) != node5->output(0) || node7->input(0) != node2->output(0) ||
          node8->input(0) != node7->output(0) || node9->input(0) != node8->output(0) ||
          node10->input(0) != node4->output(0) || node11->input(0) != node6->output(0) ||
          node12->input(0) != node11->output(0) || node13->input(0) != node10->output(0) ||
          node14->input(0) != node9->output(0) || node14->input(1) != node13->output(0) ||
          node15->input(0) != node14->output(0) || node16->input(0) != node15->output(0) ||
          node16->input(1) != node12->output(0) || node17->input(0) != node16->output(0) ||
          node18->input(0) != node17->output(0) || node19->input(0) != node18->output(0) ||
          node20->input(0) != node19->output(0))
        continue;

      std::vector<float> q_B = get_node_attr_from_input_af(weights[node2->input(1)]);
      std::vector<float> k_B = get_node_attr_from_input_af(weights[node4->input(1)]);
      std::vector<float> v_B = get_node_attr_from_input_af(weights[node6->input(1)]);
      std::vector<float> o_B = get_node_attr_from_input_af(weights[node20->input(1)]);

      if (q_B.size() != k_B.size() || q_B.size() != v_B.size() || q_B.size() != o_B.size())
        continue;

      int embed_dim = q_B.size();

      // 1 0 2
      std::vector<int> perm9 = get_node_attr_ai(*node9, "perm");
      std::vector<int> perm12 = get_node_attr_ai(*node12, "perm");
      if (perm9.size() != 3 || perm12.size() != 3) continue;

      if (perm9[0] != 1 || perm9[1] != 0 || perm9[2] != 2 || perm12[0] != 1 || perm12[1] != 0 ||
          perm12[2] != 2)
        continue;

      // 1 2 0
      std::vector<int> perm13 = get_node_attr_ai(*node13, "perm");
      if (perm13.size() != 3) continue;

      if (perm13[0] != 1 || perm13[1] != 2 || perm13[2] != 0) continue;

      // 1 0 2
      std::vector<int> perm17 = get_node_attr_ai(*node17, "perm");
      if (perm17.size() != 3) continue;

      if (perm17[0] != 1 || perm17[1] != 0 || perm17[2] != 2) continue;

      int softmax_axis = get_node_attr_i(*node15, "axis");
      if (softmax_axis != 2) continue;

      // 1/-1, seqlen * num_heads, embed_dim / num_heads
      std::vector<int> shape8;
      std::vector<int> shape10;
      std::vector<int> shape11;
      if (node8->input_size() == 1) {
        shape8 = get_node_attr_ai(*node8, "shape");
      } else {
        // skip weight reshape
        if (weights.find(node8->input(1)) == weights.end()) continue;

        shape8 = get_node_attr_from_input_ai(weights[node8->input(1)]);
      }
      if (node10->input_size() == 1) {
        shape10 = get_node_attr_ai(*node10, "shape");
      } else {
        // skip weight reshape
        if (weights.find(node10->input(1)) == weights.end()) continue;

        shape10 = get_node_attr_from_input_ai(weights[node10->input(1)]);
      }
      if (node11->input_size() == 1) {
        shape11 = get_node_attr_ai(*node11, "shape");
      } else {
        // skip weight reshape
        if (weights.find(node11->input(1)) == weights.end()) continue;

        shape11 = get_node_attr_from_input_ai(weights[node11->input(1)]);
      }

      if (shape8.size() != 3 || shape10.size() != 3 || shape11.size() != 3) continue;

      if (shape8[1] != shape10[1] || shape8[1] != shape11[1] || shape8[2] != shape10[2] ||
          shape8[2] != shape11[2])
        continue;

      int num_heads = embed_dim / shape8[2];

      // 1, seqlen, embed_dim
      std::vector<int> shape18;
      if (node18->input_size() == 1) {
        shape18 = get_node_attr_ai(*node18, "shape");
      } else {
        // skip weight reshape
        if (weights.find(node18->input(1)) == weights.end()) continue;

        shape18 = get_node_attr_from_input_ai(weights[node18->input(1)]);
      }

      if (shape18.size() != 3) continue;

      if (shape18[2] != embed_dim || shape18[1] * num_heads != shape8[1]) continue;

      // reduce
      node->set_op_type("noop_reducedncnn");
      node2->set_op_type("noop_reducedncnn");
      node3->set_op_type("noop_reducedncnn");
      node4->set_op_type("noop_reducedncnn");
      node5->set_op_type("noop_reducedncnn");
      node6->set_op_type("noop_reducedncnn");
      node7->set_op_type("noop_reducedncnn");
      node8->set_op_type("noop_reducedncnn");
      node9->set_op_type("noop_reducedncnn");
      node10->set_op_type("noop_reducedncnn");
      node11->set_op_type("noop_reducedncnn");
      node12->set_op_type("noop_reducedncnn");
      node13->set_op_type("noop_reducedncnn");
      node14->set_op_type("noop_reducedncnn");
      node15->set_op_type("noop_reducedncnn");
      node16->set_op_type("noop_reducedncnn");
      node17->set_op_type("noop_reducedncnn");
      node18->set_op_type("noop_reducedncnn");
      node19->set_op_type("noop_reducedncnn");

      node_reference[node2->input(0)] -= 1;
      node_reference[node4->input(0)] -= 1;
      node_reference[node6->input(0)] -= 1;
      node_reference[node7->input(0)] -= 1;
      node_reference[node7->input(1)] -= 1;
      node_reference[node8->input(0)] -= 1;
      if (node8->input_size() == 2) {
        node_reference[node8->input(1)] -= 1;
      }
      node_reference[node9->input(0)] -= 1;
      node_reference[node10->input(0)] -= 1;
      if (node10->input_size() == 2) {
        node_reference[node10->input(1)] -= 1;
      }
      node_reference[node11->input(0)] -= 1;
      if (node11->input_size() == 2) {
        node_reference[node11->input(1)] -= 1;
      }
      node_reference[node12->input(0)] -= 1;
      node_reference[node13->input(0)] -= 1;
      node_reference[node14->input(0)] -= 1;
      node_reference[node14->input(1)] -= 1;
      node_reference[node15->input(0)] -= 1;
      node_reference[node16->input(0)] -= 1;
      node_reference[node16->input(1)] -= 1;
      node_reference[node17->input(0)] -= 1;
      node_reference[node18->input(0)] -= 1;
      if (node18->input_size() == 2) {
        node_reference[node18->input(1)] -= 1;
      }
      node_reference[node19->input(0)] -= 1;
      node_reference[node20->input(0)] -= 1;

      blob_names.erase(node->output(0));
      blob_names.erase(node2->output(0));
      blob_names.erase(node3->output(0));
      blob_names.erase(node4->output(0));
      blob_names.erase(node5->output(0));
      blob_names.erase(node6->output(0));
      blob_names.erase(node7->output(0));
      blob_names.erase(node8->output(0));
      blob_names.erase(node9->output(0));
      blob_names.erase(node10->output(0));
      blob_names.erase(node11->output(0));
      blob_names.erase(node12->output(0));
      blob_names.erase(node13->output(0));
      blob_names.erase(node14->output(0));
      blob_names.erase(node15->output(0));
      blob_names.erase(node16->output(0));
      blob_names.erase(node17->output(0));
      blob_names.erase(node18->output(0));
      blob_names.erase(node19->output(0));

      std::string qw = node->input(1);
      std::string qb = node2->input(1);
      std::string kw = node3->input(1);
      std::string kb = node4->input(1);
      std::string vw = node5->input(1);
      std::string vb = node6->input(1);
      std::string ow = node19->input(1);
      std::string ob = node20->input(1);

      node20->set_op_type("MultiHeadAttention");
      node20->clear_input();
      node20->add_input(node->input(0));
      node20->add_input(node3->input(0));
      node20->add_input(node5->input(0));
      // q
      node20->add_input(qw);
      node20->add_input(qb);
      // k
      node20->add_input(kw);
      node20->add_input(kb);
      // v
      node20->add_input(vw);
      node20->add_input(vb);
      // out linear
      node20->add_input(ow);
      node20->add_input(ob);

      onnx::AttributeProto* attr_embed_dim = node20->add_attribute();
      attr_embed_dim->set_name("embed_dim");
      attr_embed_dim->set_i(embed_dim);

      onnx::AttributeProto* attr_num_heads = node20->add_attribute();
      attr_num_heads->set_name("num_heads");
      attr_num_heads->set_i(num_heads);

      reduced_node_count += 19;
      i += 19;
    }
  }

  for (int i = 0; i < node_count; i++) {
    onnx::NodeProto* node = mutable_graph->mutable_node(i);

    // MultiHeadAttention <= MatMul(qkv) - Add - Split
    //                      - Mul
    //                      - Reshape - Transpose
    //                      - Reshape - Reshape - Transpose - Transpose
    //                      - Gemm - Softmax - Gemm - Transpose - Reshape -
    //                      MatMul - Add
    if (node->op_type() == "MatMul") {
      if (i + 16 >= node_count) continue;

      if (node_reference[node->output(0)] != 1) continue;

      onnx::NodeProto* node2 = mutable_graph->mutable_node(i + 1);
      onnx::NodeProto* node3 = mutable_graph->mutable_node(i + 2);
      onnx::NodeProto* node4 = mutable_graph->mutable_node(i + 3);
      onnx::NodeProto* node5 = mutable_graph->mutable_node(i + 4);
      onnx::NodeProto* node6 = mutable_graph->mutable_node(i + 5);
      onnx::NodeProto* node7 = mutable_graph->mutable_node(i + 6);
      onnx::NodeProto* node8 = mutable_graph->mutable_node(i + 7);
      onnx::NodeProto* node9 = mutable_graph->mutable_node(i + 8);
      onnx::NodeProto* node10 = mutable_graph->mutable_node(i + 9);
      onnx::NodeProto* node11 = mutable_graph->mutable_node(i + 10);
      onnx::NodeProto* node12 = mutable_graph->mutable_node(i + 11);
      onnx::NodeProto* node13 = mutable_graph->mutable_node(i + 12);
      onnx::NodeProto* node14 = mutable_graph->mutable_node(i + 13);
      onnx::NodeProto* node15 = mutable_graph->mutable_node(i + 14);
      onnx::NodeProto* node16 = mutable_graph->mutable_node(i + 15);
      onnx::NodeProto* node17 = mutable_graph->mutable_node(i + 16);

      if (node2->op_type() != "Add" || node3->op_type() != "Split" || node4->op_type() != "Mul" ||
          node5->op_type() != "Reshape" || node6->op_type() != "Transpose" ||
          node7->op_type() != "Reshape" || node8->op_type() != "Reshape" ||
          node9->op_type() != "Transpose" || node10->op_type() != "Transpose" ||
          node11->op_type() != "MatMul" || node12->op_type() != "Softmax" ||
          node13->op_type() != "MatMul" || node14->op_type() != "Transpose" ||
          node15->op_type() != "Reshape" || node16->op_type() != "MatMul" ||
          node17->op_type() != "Add")
        continue;

      if (node_reference[node2->output(0)] != 1 || node_reference[node3->output(0)] != 1 ||
          node_reference[node3->output(1)] != 1 || node_reference[node3->output(2)] != 1 ||
          node_reference[node4->output(0)] != 1 || node_reference[node5->output(0)] != 1 ||
          node_reference[node6->output(0)] != 1 || node_reference[node7->output(0)] != 1 ||
          node_reference[node8->output(0)] != 1 || node_reference[node9->output(0)] != 1 ||
          node_reference[node10->output(0)] != 1 || node_reference[node11->output(0)] != 1 ||
          node_reference[node12->output(0)] != 1 || node_reference[node13->output(0)] != 1 ||
          node_reference[node14->output(0)] != 1 || node_reference[node15->output(0)] != 1 ||
          node_reference[node16->output(0)] != 1)
        continue;

      if (node2->input(0) != node->output(0) || node3->input(0) != node2->output(0) ||
          node4->input(0) != node3->output(0) || node5->input(0) != node4->output(0) ||
          node6->input(0) != node5->output(0) || node7->input(0) != node3->output(1) ||
          node8->input(0) != node3->output(2) || node9->input(0) != node8->output(0) ||
          node10->input(0) != node7->output(0) || node11->input(0) != node6->output(0) ||
          node11->input(1) != node10->output(0) || node12->input(0) != node11->output(0) ||
          node13->input(0) != node12->output(0) || node13->input(1) != node9->output(0) ||
          node14->input(0) != node13->output(0) || node15->input(0) != node14->output(0) ||
          node16->input(0) != node15->output(0) || node17->input(0) != node16->output(0))
        continue;

      std::vector<float> qkv_B = get_node_attr_from_input_af(weights[node2->input(1)]);
      std::vector<float> o_B = get_node_attr_from_input_af(weights[node17->input(1)]);

      if (qkv_B.size() != o_B.size() * 3) continue;

      int embed_dim = o_B.size();

      // 1 0 2
      std::vector<int> perm6 = get_node_attr_ai(*node6, "perm");
      std::vector<int> perm9 = get_node_attr_ai(*node9, "perm");
      if (perm6.size() != 3 || perm9.size() != 3) continue;

      if (perm6[0] != 1 || perm6[1] != 0 || perm6[2] != 2 || perm9[0] != 1 || perm9[1] != 0 ||
          perm9[2] != 2)
        continue;

      // 1 2 0
      std::vector<int> perm10 = get_node_attr_ai(*node10, "perm");
      if (perm10.size() != 3) continue;

      if (perm10[0] != 1 || perm10[1] != 2 || perm10[2] != 0) continue;

      // 1 0 2
      std::vector<int> perm14 = get_node_attr_ai(*node14, "perm");
      if (perm14.size() != 3) continue;

      if (perm14[0] != 1 || perm14[1] != 0 || perm14[2] != 2) continue;

      int softmax_axis = get_node_attr_i(*node12, "axis");
      if (softmax_axis != 2) continue;

      // 1/-1, seqlen * num_heads, embed_dim / num_heads
      std::vector<int> shape5;
      std::vector<int> shape7;
      std::vector<int> shape8;
      if (node5->input_size() == 1) {
        shape5 = get_node_attr_ai(*node5, "shape");
      } else {
        // skip weight reshape
        if (weights.find(node5->input(1)) == weights.end()) continue;

        shape5 = get_node_attr_from_input_ai(weights[node5->input(1)]);
      }
      if (node7->input_size() == 1) {
        shape7 = get_node_attr_ai(*node7, "shape");
      } else {
        // skip weight reshape
        if (weights.find(node7->input(1)) == weights.end()) continue;

        shape7 = get_node_attr_from_input_ai(weights[node7->input(1)]);
      }
      if (node8->input_size() == 1) {
        shape8 = get_node_attr_ai(*node8, "shape");
      } else {
        // skip weight reshape
        if (weights.find(node8->input(1)) == weights.end()) continue;

        shape8 = get_node_attr_from_input_ai(weights[node8->input(1)]);
      }

      if (shape5.size() != 3 || shape7.size() != 3 || shape8.size() != 3) continue;

      if (shape5[1] != shape7[1] || shape5[1] != shape8[1] || shape5[2] != shape7[2] ||
          shape5[2] != shape8[2])
        continue;

      int num_heads = embed_dim / shape5[2];

      // 1, seqlen, embed_dim
      std::vector<int> shape15;
      if (node15->input_size() == 1) {
        shape15 = get_node_attr_ai(*node15, "shape");
      } else {
        // skip weight reshape
        if (weights.find(node15->input(1)) == weights.end()) continue;

        shape15 = get_node_attr_from_input_ai(weights[node15->input(1)]);
      }

      if (shape15.size() != 3) continue;

      if (shape15[2] != embed_dim || shape15[1] * num_heads != shape8[1]) continue;

      // reduce
      node->set_op_type("noop_reducedncnn");
      node2->set_op_type("noop_reducedncnn");
      node3->set_op_type("noop_reducedncnn");
      node4->set_op_type("noop_reducedncnn");
      node5->set_op_type("noop_reducedncnn");
      node6->set_op_type("noop_reducedncnn");
      node7->set_op_type("noop_reducedncnn");
      node8->set_op_type("noop_reducedncnn");
      node9->set_op_type("noop_reducedncnn");
      node10->set_op_type("noop_reducedncnn");
      node11->set_op_type("noop_reducedncnn");
      node12->set_op_type("noop_reducedncnn");
      node13->set_op_type("noop_reducedncnn");
      node14->set_op_type("noop_reducedncnn");
      node15->set_op_type("noop_reducedncnn");
      node16->set_op_type("noop_reducedncnn");

      node_reference[node2->input(0)] -= 1;
      node_reference[node3->input(0)] -= 1;
      node_reference[node4->input(0)] -= 1;
      node_reference[node4->input(1)] -= 1;
      node_reference[node5->input(0)] -= 1;
      if (node5->input_size() == 2) {
        node_reference[node5->input(1)] -= 1;
      }
      node_reference[node6->input(0)] -= 1;
      node_reference[node7->input(0)] -= 1;
      if (node7->input_size() == 2) {
        node_reference[node7->input(1)] -= 1;
      }
      node_reference[node8->input(0)] -= 1;
      if (node8->input_size() == 2) {
        node_reference[node8->input(1)] -= 1;
      }
      node_reference[node9->input(0)] -= 1;
      node_reference[node10->input(0)] -= 1;
      node_reference[node11->input(0)] -= 1;
      node_reference[node11->input(1)] -= 1;
      node_reference[node12->input(0)] -= 1;
      node_reference[node13->input(0)] -= 1;
      node_reference[node13->input(1)] -= 1;
      node_reference[node14->input(0)] -= 1;
      node_reference[node15->input(0)] -= 1;
      if (node15->input_size() == 2) {
        node_reference[node15->input(1)] -= 1;
      }
      node_reference[node16->input(0)] -= 1;
      node_reference[node17->input(0)] -= 1;

      blob_names.erase(node->output(0));
      blob_names.erase(node2->output(0));
      blob_names.erase(node3->output(0));
      blob_names.erase(node3->output(1));
      blob_names.erase(node3->output(2));
      blob_names.erase(node4->output(0));
      blob_names.erase(node5->output(0));
      blob_names.erase(node6->output(0));
      blob_names.erase(node7->output(0));
      blob_names.erase(node8->output(0));
      blob_names.erase(node9->output(0));
      blob_names.erase(node10->output(0));
      blob_names.erase(node11->output(0));
      blob_names.erase(node12->output(0));
      blob_names.erase(node13->output(0));
      blob_names.erase(node14->output(0));
      blob_names.erase(node15->output(0));
      blob_names.erase(node16->output(0));

      std::string qkvw = node->input(1);
      std::string qkvb = node2->input(1);
      std::string ow = node16->input(1);
      std::string ob = node17->input(1);

      node17->set_op_type("MultiHeadAttention");
      node17->clear_input();
      node17->add_input(node->input(0));
      // qkv
      node17->add_input(qkvw);
      node17->add_input(qkvb);
      // out linear
      node17->add_input(ow);
      node17->add_input(ob);

      onnx::AttributeProto* attr_embed_dim = node17->add_attribute();
      attr_embed_dim->set_name("embed_dim");
      attr_embed_dim->set_i(embed_dim);

      onnx::AttributeProto* attr_num_heads = node17->add_attribute();
      attr_num_heads->set_name("num_heads");
      attr_num_heads->set_i(num_heads);

      reduced_node_count += 16;
      i += 16;
    }
  }
}
