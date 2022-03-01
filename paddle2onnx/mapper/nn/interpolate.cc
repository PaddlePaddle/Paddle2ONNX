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

#include "paddle2onnx/mapper/nn/interpolate.h"

namespace paddle2onnx {
REGISTER_MAPPER(bilinear_interp_v2, InterpolateMapper)
REGISTER_MAPPER(nearest_interp_v2, InterpolateMapper)
REGISTER_MAPPER(bicubic_interp_v2, InterpolateMapper)
REGISTER_MAPPER(linear_interp_v2, InterpolateMapper)
REGISTER_MAPPER(trilinear_interp_v2, InterpolateMapper)

int32_t InterpolateMapper::GetMinOpset(bool verbose) {
  auto op = parser_->GetOpDesc(block_idx_, op_idx_);
  if (data_layout_ == "NHWC") {
    if (verbose) {
      std::cerr << "Paddle2ONNX: NHWC is not supported for op: " << op.type()
                << std::endl;
    }
    return -1;
  }
  auto x_info = parser_->GetOpInput(block_idx_, op_idx_, "X");
  if (x_info[0].Rank() > 5 && x_info[0].Rank() < 3) {
    if (verbose) {
      std::cerr << "Paddle2ONNX: Only support 3D/4D/5D tensor for op: "
                << op.type() << ", but now the dimension is "
                << x_info[0].Rank() << std::endl;
    }
    return -1;
  }
  return 11;
}

std::string InterpolateMapper::ComputeOutSize(OnnxHelper* helper) {
  bool has_out_size = parser_->OpHasInput(block_idx_, op_idx_, "OutSize");
  bool has_size_tensor = parser_->OpHasInput(block_idx_, op_idx_, "SizeTensor");
  if (has_out_size) {
    auto out_size_info = parser_->GetOpInput(block_idx_, op_idx_, "OutSize");
    return helper->AutoCast(out_size_info[0].name, out_size_info[0].dtype,
                            P2ODataType::INT64);
  } else {
    auto size_tensor_info =
        parser_->GetOpInput(block_idx_, op_idx_, "SizeTensor");
    return helper->ConcatIndices(size_tensor_info);
  }
}

std::string InterpolateMapper::ComputeScale(OnnxHelper* helper) {
  auto scale_info = parser_->GetOpInput(block_idx_, op_idx_, "Scale");
  auto scale = helper->AutoCast(scale_info[0].name, scale_info[0].dtype,
                                P2ODataType::FP32);
  auto padding = helper->Constant(ONNX_NAMESPACE::TensorProto::FLOAT,
                                  std::vector<float>(2, 1.0));
  scale = helper->Concat({padding, scale}, 0);
  return scale;
}

void InterpolateMapper::Opset11(OnnxHelper* helper) {
  auto x_info = parser_->GetOpInput(block_idx_, op_idx_, "X");
  auto out_info = parser_->GetOpOutput(block_idx_, op_idx_, "Out");
  auto op = parser_->GetOpDesc(block_idx_, op_idx_);
  std::string coordinate_transformation_mode = "half_pixel";
  auto resize_type = resize_mapper_[method_];
  if (align_corners_) {
    coordinate_transformation_mode = "align_corners";
  } else if (resize_type == "nearest") {
    coordinate_transformation_mode = "asymmetric";
  } else if (align_mode_ == 1 && resize_type != "cubic") {
    coordinate_transformation_mode = "asymmetric";
  }
  std::string scale = "";
  std::string size = "";
  bool has_out_size = parser_->OpHasInput(block_idx_, op_idx_, "OutSize");
  bool has_size_tensor = parser_->OpHasInput(block_idx_, op_idx_, "SizeTensor");
  bool has_scale_tensor = parser_->OpHasInput(block_idx_, op_idx_, "Scale");
  if (has_out_size || has_size_tensor) {
    size = ComputeOutSize(helper);
  } else if (has_scale_tensor) {
    scale = ComputeScale(helper);
  } else {
    // get size or scale from attribute
    if (out_d_ > 0 || out_w_ > 0 || out_h_ > 0) {
      std::vector<int64_t> out_size;
      if (x_info[0].Rank() == 5) {
        out_size.push_back(out_d_);
      }
      if (x_info[0].Rank() == 4) {
        out_size.push_back(out_h_);
      }
      out_size.push_back(out_w_);
      size = helper->Constant(ONNX_NAMESPACE::TensorProto::INT64, out_size);
    } else {
      std::vector<float> scale_;
      parser_->GetOpAttr(op, "scale", &scale_);
      float padding = 1.0;
      scale_.insert(scale_.begin(), padding);
      scale_.insert(scale_.begin(), padding);
      scale = helper->Constant(ONNX_NAMESPACE::TensorProto::FLOAT, scale_);
    }
  }
  std::string roi =
      helper->Constant(ONNX_NAMESPACE::TensorProto::FLOAT,
                       std::vector<float>(x_info[0].Rank() * 2, 1.0));
  if (scale == "") {
    // has to generate a empty tensor for resize
    scale = helper->Constant(ONNX_NAMESPACE::TensorProto::FLOAT,
                             std::vector<float>());
  }
  if (size != "") {
    auto ipt_shape = helper->MakeNode("Shape", {x_info[0].name})->output(0);
    auto nc = helper->Slice(ipt_shape, {0}, {0}, {2});
    size = helper->Concat({nc, size}, 0);
  }
  auto node = helper->MakeNode("Resize", {x_info[0].name, roi, scale, size},
                               {out_info[0].name});
  Assert(resize_mapper_.find(op.type()) != resize_mapper_.end(),
         "Cannot find " + op.type() + " in resize_mapper.");
  AddAttribute(node, "mode", resize_mapper_[op.type()]);
  AddAttribute(node, "coordinate_transformation_mode",
               coordinate_transformation_mode);
  if (resize_mapper_[op.type()] == "nearest" &&
      coordinate_transformation_mode == "asymmetric") {
    AddAttribute(node, "nearest_mode", "floor");
  }
}

}  // namespace paddle2onnx
