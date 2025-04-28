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

#include "paddle2onnx/mapper/tensor/tile.h"

namespace paddle2onnx {
REGISTER_PIR_MAPPER(tile, TileMapper)

int32_t TileMapper::GetMinOpsetVersion(bool verbose) {
  Logger(verbose, 10) << RequireOpset(10) << std::endl;
  return 10;
}

void TileMapper::Opset10() {
  Assert(HasInput("repeat_times"),
         "Tile operator must has repeat_times input.");
  auto x_info = GetInput("x");
  auto out_info = GetOutput("out");
  auto repeats_info = GetInput("repeat_times");
  auto repeats = helper_->ConcatIndices(repeats_info);
  if (x_info[0].Rank() == 0) {
    auto unsqueeze = helper_->Unsqueeze(x_info[0].name, {0});
    helper_->MakeNode("Tile", {unsqueeze, repeats}, {out_info[0].name});
  } else {
    auto ones = helper_->Constant(GetOnnxDtype(P2ODataType::INT64),
                                  std::vector<int64_t>(x_info[0].Rank(), 1));
    auto x_rank = helper_->Constant(GetOnnxDtype(P2ODataType::INT64),
                                    std::vector<int64_t>{x_info[0].Rank()});
    auto repeats_shape = helper_->MakeNode("Shape", {repeats})->output(0);
    auto diff = helper_->MakeNode("Sub", {x_rank, repeats_shape})->output(0);
    auto zero = helper_->Constant(GetOnnxDtype(P2ODataType::INT64),
                                  std::vector<int64_t>{0});
    auto fp64_diff =
        helper_->AutoCast(diff, P2ODataType::INT64, P2ODataType::FP64);
    auto fp64_zero = helper_->Constant(GetOnnxDtype(P2ODataType::FP64),
                                       std::vector<double>{0.0});
    auto need_expand =
        helper_->MakeNode("Max", {fp64_diff, fp64_zero})->output(0);
    need_expand =
        helper_->AutoCast(need_expand, P2ODataType::FP64, P2ODataType::INT64);
    auto expand_repeats =
        helper_->MakeNode("Slice", {ones, zero, need_expand, zero})->output(0);
    repeats = helper_->Concat({expand_repeats, repeats}, 0);
    helper_->MakeNode("Tile", {x_info[0].name, repeats}, {out_info[0].name});
  }
}

}  // namespace paddle2onnx
