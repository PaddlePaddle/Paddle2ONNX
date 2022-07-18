//   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// before Fusion:
//                   [Input](BxSxW)
//                         |
//                 LayerNormalization
//             /       |        |     \     [Weights](WxW')
//            /        |        |      \    /
//           |   q_MatMul    k_MatMul  v_MatMul  [Bias](W')
//           |         |        |        |    /
//           |     q_Add     k_Add     v_Add     [Shape=0,0,N,H]
//           |         |        |        |      /
//           | q_Reshape   k_Reshape   v_Reshape
//           |         |        |        |
//           |q_Transpose  k_Transpose v_Transpose
//           |  (0,2,1,3)  (0,2,3,1)    (perm=0,2,1,3)
//           |         \       /         |
//           |      qk_MatMul            |
//           |           |               |
//           |           |               |
//           |        qk_Mul             |
//           |            \              |
//           |       mask_Add <---------/----------------extra_add_qk(BxNxSxS)
//           |             |           /
//           |          Softmax       /
//           |             \         /
//           |              \       /
//           |            qkv_MatMul
//           |                   |
//           |                Transpose (perm=0,2,1,3)
//           |                   |
//           |                Reshape---[shape=0,0,W']
//           |                   |
//           |                 MatMul----[Weights](W'xW)
//           |                   |
//           |                  Add----[Bias](W)
//           +-------------------|---+
//                               |   |
//                                Add

// After Fusion:
//   LayerNormalization  [Weights](Wx3W')
//       |        \      /   [Bias](3W')
//       |         \    /   /
//       |         Attention <------------extra_add_qk(BxNxSxS)
//       \          |
//        \        MatMul
//         \        |
//          \      Add
//           +------|---+
//                  |   |
//                   Add

#include <numeric>

#include "onnx/defs/tensor_util.h"
#include "onnxoptimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct FuseAttention final : public PredicateBasedPass {
  explicit FuseAttention()
      : PredicateBasedPass(PassType::Fuse, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override { return "fuse_attention"; }

  bool patternMatchPredicate(Node *node) override {
    bool q_sucess = false;
    bool k_sucess = false;
    bool v_sucess = false;
    Node *qkv_matmul;
    Node *qk_matmul;
    // find qkv matmul
    if (node->kind() == kReshape &&
        node->inputs()[0]->node()->kind() == kTranspose &&
        node->inputs()[0]->node()->inputs()[0]->node()->kind() == kMatMul) {
      qkv_matmul = node->inputs()[0]->node()->inputs()[0]->node();
    } else {
      return false;
    }
    // Find v path nodes: v_transpose -> v_reshape -> v_add -> v_matmul
    if (qkv_matmul->inputs()[1]->node()->kind() == kTranspose &&
        qkv_matmul->inputs()[1]->node()->inputs()[0]->node()->kind() ==
            kReshape &&
        qkv_matmul->inputs()[1]
                ->node()
                ->inputs()[0]
                ->node()
                ->inputs()[0]
                ->node()
                ->kind() == kAdd &&
        qkv_matmul->inputs()[1]
                ->node()
                ->inputs()[0]
                ->node()
                ->inputs()[0]
                ->node()
                ->inputs()[0]
                ->node()
                ->kind() == kMatMul) {
      v_sucess = true;
    } else {
      return false;
    }
    // Find mask_Add path nodes: softmax -> mask_Add ->  -> qk_mul -> qk_matmul
    if (qkv_matmul->inputs()[0]->node()->kind() == kSoftmax &&
        qkv_matmul->inputs()[0]->node()->inputs()[0]->node()->kind() == kAdd &&
        qkv_matmul->inputs()[0]
                ->node()
                ->inputs()[0]
                ->node()
                ->inputs()[0]
                ->node()
                ->kind() == kMul &&
        qkv_matmul->inputs()[0]
                ->node()
                ->inputs()[0]
                ->node()
                ->inputs()[0]
                ->node()
                ->inputs()[0]
                ->node()
                ->kind() == kMatMul) {
      qk_matmul = qkv_matmul->inputs()[0]
                      ->node()
                      ->inputs()[0]
                      ->node()
                      ->inputs()[0]
                      ->node()
                      ->inputs()[0]
                      ->node();
    } else {
      return false;
    }
    // Find q path nodes: q_transpose -> q_reshape -> q_add -> q_matmul
    if (qk_matmul->inputs()[0]->node()->kind() == kTranspose &&
        qk_matmul->inputs()[0]->node()->inputs()[0]->node()->kind() ==
            kReshape &&
        qk_matmul->inputs()[0]
                ->node()
                ->inputs()[0]
                ->node()
                ->inputs()[0]
                ->node()
                ->kind() == kAdd &&
        qk_matmul->inputs()[0]
                ->node()
                ->inputs()[0]
                ->node()
                ->inputs()[0]
                ->node()
                ->inputs()[0]
                ->node()
                ->kind() == kMatMul) {
      q_sucess = true;
    } else {
      return false;
    }
    // Find k path nodes: k_transpose -> k_reshape -> k_add -> k_matmul
    if (qk_matmul->inputs()[1]->node()->kind() == kTranspose &&
        qk_matmul->inputs()[1]->node()->inputs()[0]->node()->kind() ==
            kReshape &&
        qk_matmul->inputs()[1]
                ->node()
                ->inputs()[0]
                ->node()
                ->inputs()[0]
                ->node()
                ->kind() == kAdd &&
        qk_matmul->inputs()[1]
                ->node()
                ->inputs()[0]
                ->node()
                ->inputs()[0]
                ->node()
                ->inputs()[0]
                ->node()
                ->kind() == kMatMul) {
      k_sucess = true;
    } else {
      return false;
    }

    return q_sucess && k_sucess && v_sucess;
  }

  bool runTransform(Node *n, Graph &graph,
                    NodeDestroyType &destroy_current) override {
    Node *transpose = n->inputs()[0]->node();
    Node *qkv_matmul = transpose->inputs()[0]->node();
    Node *softmax = qkv_matmul->inputs()[0]->node();
    Node *mask_add = softmax->inputs()[0]->node();
    Node *mul = mask_add->inputs()[0]->node();
    Node *qk_matmul = mul->inputs()[0]->node();
    // q path
    Node *q_transpose = qk_matmul->inputs()[0]->node();
    Node *q_reshape = q_transpose->inputs()[0]->node();
    Node *q_add = q_reshape->inputs()[0]->node();
    Node *q_matmul = q_add->inputs()[0]->node();
    // k path
    Node *k_transpose = qk_matmul->inputs()[1]->node();
    Node *k_reshape = k_transpose->inputs()[0]->node();
    Node *k_add = k_reshape->inputs()[0]->node();
    Node *k_matmul = k_add->inputs()[0]->node();
    // v path
    Node *v_transpose = qkv_matmul->inputs()[1]->node();
    Node *v_reshape = v_transpose->inputs()[0]->node();
    Node *v_add = v_reshape->inputs()[0]->node();
    Node *v_matmul = v_add->inputs()[0]->node();
    // input node
    Node *source_node = q_matmul->inputs()[0]->node();

    // Obtain Q, K ,V weights and bias
    Node *q_weight_node = q_matmul->inputs()[1]->node();
    if (q_weight_node->kind() != kConstant) {
      return false;
    }
    Tensor q_weight = q_weight_node->t(kvalue);
    if (q_weight.sizes().size() != 2) {
      return false;
    }
    Node *q_bias_node = q_add->inputs()[1]->node();
    if (q_bias_node->kind() != kConstant) {
      return false;
    }
    Tensor q_bias = q_bias_node->t(kvalue);
    if (q_bias.sizes().size() != 1) {
      return false;
    }
    Node *k_weight_node = k_matmul->inputs()[1]->node();
    if (k_weight_node->kind() != kConstant) {
      return false;
    }
    Tensor k_weight = k_weight_node->t(kvalue);
    if (k_weight.sizes().size() != 2) {
      return false;
    }
    Node *k_bias_node = k_add->inputs()[1]->node();
    if (k_bias_node->kind() != kConstant) {
      return false;
    }
    Tensor k_bias = k_bias_node->t(kvalue);
    if (k_bias.sizes().size() != 1) {
      return false;
    }
    Node *v_weight_node = v_matmul->inputs()[1]->node();
    if (v_weight_node->kind() != kConstant) {
      return false;
    }
    Tensor v_weight = v_weight_node->t(kvalue);
    if (v_weight.sizes().size() != 2) {
      return false;
    }
    Node *v_bias_node = v_add->inputs()[1]->node();
    if (v_bias_node->kind() != kConstant) {
      return false;
    }
    Tensor v_bias = v_bias_node->t(kvalue);
    if (v_bias.sizes().size() != 1) {
      return false;
    }
    // Merge Q, K and V weights and bias
    std::vector<float> qkv_weights(q_weight.sizes()[0] * q_weight.sizes()[1] *
                                   3);
    std::vector<float> qkv_bias(q_bias.sizes()[0] * 3);
    std::vector<std::vector<float>> w_vec = {ParseData<float>(&q_weight),
                                             ParseData<float>(&k_weight),
                                             ParseData<float>(&v_weight)};
    std::vector<std::vector<float>> v_vec = {ParseData<float>(&q_bias),
                                             ParseData<float>(&k_bias),
                                             ParseData<float>(&v_bias)};
    int64_t dims_h = q_weight.sizes()[0];
    int64_t dims_w = q_weight.sizes()[1];
    // combine Q, K and V weights into qkv_weights
    for (int i = 0; i < dims_h; i++) {
      for (int j = 0; j < 3; j++) {
        for (int k = 0; k < dims_w; k++) {
          int out_index = i * (3 * dims_w) + j * dims_w + k;
          int in_index = i * dims_w + k;
          qkv_weights[out_index] = w_vec[j][in_index];
        }
      }
    }
    // combine Q, K and V bias into qkv_bias
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < dims_w; j++) {
        qkv_bias[i * dims_w + j] = v_vec[i][j];
      }
    }
    // Create Attention Node
    Node *attention_node = graph.create(Symbol("Attention"), 1);
    Tensor attention_qkv_weights, attention_qkv_bias;
    attention_qkv_weights.sizes().push_back(q_weight.sizes()[0]);
    attention_qkv_weights.sizes().push_back(q_weight.sizes()[1] * 3);
    attention_qkv_weights.floats() = qkv_weights;
    attention_qkv_weights.elem_type() = TensorProto_DataType_FLOAT;
    attention_qkv_bias.sizes().push_back(qkv_bias.size());
    attention_qkv_bias.floats() = qkv_bias;
    attention_qkv_bias.elem_type() = TensorProto_DataType_FLOAT;
    // Construct qkv_weight constant input node
    Node *attention_inputs_qkv_weight = graph.create(kConstant, 1);
    attention_inputs_qkv_weight->t_(Symbol("value"), attention_qkv_weights);
    std::vector<Dimension> s_qkv_weight = {q_weight.sizes()[0],
                                           q_weight.sizes()[1] * 3};
    attention_inputs_qkv_weight->output()->setSizes(s_qkv_weight);
    attention_inputs_qkv_weight->output()->setElemType(
        TensorProto_DataType_FLOAT);
    attention_inputs_qkv_weight->insertAfter(source_node);
    // Construct qkv_bias constant input node
    Node *attention_inputs_qkv_bias = graph.create(kConstant, 1);
    attention_inputs_qkv_bias->t_(Symbol("value"), attention_qkv_bias);
    std::vector<Dimension> s_qkv_bias = {q_weight.sizes()[1] * 3};
    attention_inputs_qkv_bias->output()->setSizes(s_qkv_bias);
    attention_inputs_qkv_bias->output()->setElemType(
        TensorProto_DataType_FLOAT);
    attention_inputs_qkv_bias->insertAfter(source_node);
    // Construct Attention inputs
    int num_heads = q_weight.sizes()[1] / 64;
    Value *attention_output = n->output();
    attention_node->i_(Symbol("num_heads"), num_heads);
    attention_node->i_(Symbol("unidirectional"), 0);
    attention_node->addInput(source_node->output());
    attention_node->addInput(attention_inputs_qkv_weight->output());
    attention_node->addInput(attention_inputs_qkv_bias->output());
    // Add two optional Node for extra_add_qk
    auto *attention_inputs_mask_index = graph.create(kUndefined, 1);
    graph.appendNode(attention_inputs_mask_index);
    attention_inputs_mask_index->outputs()[0]->setUniqueName("");
    auto *attention_inputs_past = graph.create(kUndefined, 1);
    graph.appendNode(attention_inputs_past);
    attention_inputs_past->outputs()[0]->setUniqueName("");

    attention_node->addInput(attention_inputs_mask_index->output());
    attention_node->addInput(attention_inputs_past->output());

    // Add tile node to solve the problem of unable to broadcast automatically
    // Get the sequence length dimension
    Node *shape = graph.create(Symbol("Shape"), 1);
    shape->addInput(source_node->output());
    shape->insertAfter(source_node);
    Node *gather = graph.create(Symbol("Gather"), 1);
    // add indices for gather
    Node *indices = graph.create(kConstant, 1);
    Tensor indice_value;
    indice_value.sizes().push_back(static_cast<int64_t>(1));
    std::vector<int64_t> value = {1};
    indice_value.int64s() = value;
    indice_value.elem_type() = TensorProto_DataType_INT64;
    indices->t_(Symbol("value"), indice_value);
    std::vector<Dimension> s_indice = {1};
    indices->output()->setSizes(s_indice);
    indices->output()->setElemType(TensorProto_DataType_INT64);
    indices->insertAfter(shape);
    gather->addInput(shape->output());
    gather->addInput(indices->output());
    gather->insertAfter(shape);
    // Construct the Shape of the Tile by Concat
    Node *constant0 = graph.create(kConstant, 1);
    Tensor t0;
    t0.sizes().push_back(static_cast<int64_t>(1));
    std::vector<int64_t> shape_value0 = {1};
    t0.int64s() = shape_value0;
    t0.elem_type() = TensorProto_DataType_INT64;
    constant0->t_(Symbol("value"), t0);
    std::vector<Dimension> s0 = {1};
    constant0->output()->setSizes(s0);
    constant0->output()->setElemType(TensorProto_DataType_INT64);
    constant0->insertAfter(gather);

    Node *constant1 = graph.create(kConstant, 1);
    Tensor t1;
    t1.sizes().push_back(static_cast<int64_t>(1));
    std::vector<int64_t> shape_value1 = {num_heads};
    t1.int64s() = shape_value1;
    t1.elem_type() = TensorProto_DataType_INT64;
    constant1->t_(Symbol("value"), t1);
    std::vector<Dimension> s1 = {1};
    constant1->output()->setSizes(s1);
    constant1->output()->setElemType(TensorProto_DataType_INT64);
    constant1->insertAfter(gather);

    Node *constant2 = graph.create(kConstant, 1);
    Tensor t2;
    t2.sizes().push_back(static_cast<int64_t>(1));
    std::vector<int64_t> shape_value2 = {1};
    t2.int64s() = shape_value2;
    t2.elem_type() = TensorProto_DataType_INT64;
    constant2->t_(Symbol("value"), t2);
    std::vector<Dimension> s2 = {1};
    constant2->output()->setSizes(s2);
    constant2->output()->setElemType(TensorProto_DataType_INT64);
    constant2->insertAfter(gather);

    Node *concat = graph.create(kConcat, 1);
    concat->i_(kaxis, 0);
    concat->addInput(constant0->output());
    concat->addInput(constant1->output());
    concat->addInput(gather->output());
    concat->addInput(constant2->output());
    concat->insertAfter(gather);
    // Add Tile before mask_add
    Node *tile = graph.create(kTile, 1);
    tile->addInput(mask_add->inputs()[1]);
    tile->addInput(concat->output());
    tile->insertBefore(mask_add);
    attention_node->addInput(tile->output());
    attention_node->insertBefore(q_matmul);
    attention_node->output()->setSizes(attention_output->sizes());
    attention_node->output()->setElemType(attention_output->elemType());
    attention_node->setDomain("com.microsoft");
    const bool replacing_success = tryReplacingAllUsesWith(n, attention_node);
    if (!replacing_success) {
      return false;
    }
    // remove useless node
    if (!tryReplacingAllUsesWith(transpose->output(), transpose->inputs()[0])) {
      return false;
    }
    if (!tryReplacingAllUsesWith(qkv_matmul->output(),
                                 qkv_matmul->inputs()[0])) {
      return false;
    }
    if (!tryReplacingAllUsesWith(softmax->output(), softmax->inputs()[0])) {
      return false;
    }
    if (!tryReplacingAllUsesWith(mask_add->output(), mask_add->inputs()[0])) {
      return false;
    }
    if (!tryReplacingAllUsesWith(mul->output(), mul->inputs()[0])) {
      return false;
    }
    if (!tryReplacingAllUsesWith(qk_matmul->output(), qk_matmul->inputs()[0])) {
      return false;
    }
    if (!tryReplacingAllUsesWith(q_transpose->output(),
                                 q_transpose->inputs()[0])) {
      return false;
    }
    if (!tryReplacingAllUsesWith(q_reshape->output(), q_reshape->inputs()[0])) {
      return false;
    }
    if (!tryReplacingAllUsesWith(q_add->output(), q_add->inputs()[0])) {
      return false;
    }
    if (!tryReplacingAllUsesWith(q_matmul->output(), q_matmul->inputs()[0])) {
      return false;
    }
    if (!tryReplacingAllUsesWith(k_transpose->output(),
                                 k_transpose->inputs()[0])) {
      return false;
    }
    if (!tryReplacingAllUsesWith(k_reshape->output(), k_reshape->inputs()[0])) {
      return false;
    }
    if (!tryReplacingAllUsesWith(k_add->output(), k_add->inputs()[0])) {
      return false;
    }
    if (!tryReplacingAllUsesWith(k_matmul->output(), k_matmul->inputs()[0])) {
      return false;
    }
    if (!tryReplacingAllUsesWith(v_transpose->output(),
                                 v_transpose->inputs()[0])) {
      return false;
    }
    if (!tryReplacingAllUsesWith(v_reshape->output(), v_reshape->inputs()[0])) {
      return false;
    }
    if (!tryReplacingAllUsesWith(v_add->output(), v_add->inputs()[0])) {
      return false;
    }
    if (!tryReplacingAllUsesWith(v_matmul->output(), v_matmul->inputs()[0])) {
      return false;
    }

    return true;
  }
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
