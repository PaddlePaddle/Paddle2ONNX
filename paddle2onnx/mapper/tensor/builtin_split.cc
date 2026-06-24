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

#include "paddle2onnx/mapper/tensor/builtin_split.h"

#include <stdexcept>

namespace paddle2onnx {
REGISTER_PIR_MAPPER(builtin_split, BuiltinSplitMapper)

int64_t BuiltinSplitMapper::GetOutputNum() {
  auto& op = if_in_cf_block ? pir_parser_->sub_blocks_ops[pir_op_idx_]
                            : pir_parser_->global_blocks_ops[pir_op_idx_];
  return op->num_results();
}

bool BuiltinSplitMapper::IsEinsumOut() {
  auto& op = if_in_cf_block ? pir_parser_->sub_blocks_ops[pir_op_idx_]
                            : pir_parser_->global_blocks_ops[pir_op_idx_];
  if (op->name() != "builtin.split") {
    throw std::runtime_error(
        "The operator type must be builtin.split, but the actual "
        "operator type is " +
        op->name());
  }
  if (op->num_operands() > 0 &&
      op->operand_source(0).defining_op() &&
      op->operand_source(0).defining_op()->name() == "pd_op.einsum") {
    Warn() << "Skip builtin.split." << std::endl;
    return true;
  }
  return false;
}

void BuiltinSplitMapper::Opset7() {
  if (IsEinsumOut()) return;
  auto input_info = GetInput(0);
  int64_t output_num = GetOutputNum();
  if (output_num != static_cast<int64_t>(input_info.size())) {
    throw std::runtime_error(
        "The number of inputs and outputs must be the same, but the actual "
        "input number is " +
        std::to_string(input_info.size()) + " and output number is " +
        std::to_string(output_num));
  }
  for (int64_t i = 0; i < output_num; ++i) {
    auto output_info = GetOutput(i);
    helper_->MakeNode("Identity", {input_info[i].name}, {output_info[0].name});
  }
}

}  // namespace paddle2onnx
