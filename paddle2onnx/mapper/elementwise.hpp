// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#pragma once
#include "paddle2onnx/mapper/mapper.hpp"

namespace paddle2onnx {

class ElementwiseMapper : public Mapper {
 public:
  ElementwiseMapper(const PaddleParser& p, int64_t block_id, int64_t op_id)
      : Mapper(p, block_id, op_id) {
    auto op = parser->GetOpDesc(block_idx, op_idx);
    op_type = op.type();
    parser->GetOpAttr(op, "axis", &axis);

    op_mapper["elementwise_add"] = "Add";
  }

  bool Check() {
    auto op = parser->GetOpDesc(block_idx, op_idx);
    auto iter = op_mapper.find(op.type());
    if (op_mapper.end() == iter) {
      // TODO this log should be controled to show or now show
      std::cerr << "Cannot find " << op.type() << " in elementwise op_mapper."
                << std::endl;
      return false;
    }
    auto x_info = parser->GetOpInput(block_idx, op_idx, "X");
    if (axis < -1 or axis > x_info[0].shape.size()) {
      std::cerr << "find illegal axis in " << op.type() << "." << std::endl;
      return false;
    }
    return true;
  }

  void Opset7(std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>* nodes) {
    nodes->clear();
    auto x_info = parser->GetOpInput(block_idx, op_idx, "X");
    auto y_info = parser->GetOpInput(block_idx, op_idx, "Y");
    auto out_info = parser->GetOpInput(block_idx, op_idx, "Out");
    auto iter = op_mapper.find(op_type);
    Assert(op_mapper.end() != iter,
           "Cannot find " + op_type + " in activation op_mapper.");

    int64_t x_rank = (int64_t)(x_info[0].shape.size());
    int64_t y_rank = (int64_t)(y_info[0].shape.size());
    if (axis == -1 || axis == (x_rank - 1) || x_rank == y_rank) {
      auto node = MakeNode(iter->second, {x_info[0].name, y_info[0].name},
                           {out_info[0].name});
      nodes->push_back(node);
    } else {
      // TODO while axis == 1, and the previous node is conv, we could merge
      // this to conv as bias
      std::vector<int64_t> broadcast_shape(x_rank, 1);
      for (auto i = axis; i < axis + y_rank; ++i) {
        broadcast_shape[i] = y_info[0].shape[i - axis];
      }
      Weight shape;
      shape.set(P2ODataType::INT64, {x_rank}, broadcast_shape);
    }
  }

 private:
  std::map<std::string, std::string> op_mapper;
  std::string op_type;
  int64_t axis;
};
}  // namespace paddle2onnx
