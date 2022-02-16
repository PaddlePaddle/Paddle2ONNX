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

#include "paddle2onnx/mapper/detection/yolo_box.h"

namespace paddle2onnx {

REGISTER_MAPPER(yolo_box, YoloBoxMapper)

int32_t YoloBoxMapper::GetMinOpset(bool verbose) { return 11; }

void YoloBoxMapper::Opset11(OnnxHelper* helper) {
  auto x_info_ori = parser_->GetOpInput(block_idx_, op_idx_, "X");

  // handle the float64 input
  auto x_info = x_info_ori;
  if (x_info_ori[0].dtype != P2ODataType::FP32) {
    x_info[0].name = helper->AutoCast(x_info_ori[0].name, x_info_ori[0].dtype,
                                      P2ODataType::FP32);
    x_info[0].dtype = P2ODataType::FP32;
  }

  auto im_size_info = parser_->GetOpInput(block_idx_, op_idx_, "ImgSize");
  auto boxes_info = parser_->GetOpOutput(block_idx_, op_idx_, "Boxes");
  auto scores_info = parser_->GetOpOutput(block_idx_, op_idx_, "Scores");
  int64_t max_int = 999999;

  int64_t anchor_num = anchors_.size() / 2;

  auto x_shape = helper->MakeNode("Shape", {x_info[0].name});
  std::vector<std::string> nchw =
      helper->Split(x_shape->output(0), std::vector<int64_t>(4, 1), int64_t(0));
  std::string float_h =
      helper->AutoCast(nchw[2], P2ODataType::INT64, x_info[0].dtype);
  std::string float_w =
      helper->AutoCast(nchw[3], P2ODataType::INT64, x_info[0].dtype);

  auto anchor_num_tensor =
      helper->Constant({1}, ONNX_NAMESPACE::TensorProto::INT64, anchor_num);

  auto x_name = x_info[0].name;
  if (iou_aware_) {
    // Here we use the feature that while value is very large, it equals to the
    // ends This is a standared definition in ONNX However not sure all the
    // inference engines implements `Slice` this way Let's handle this issue
    // later
    x_name = helper->Slice(x_name, {0, 1, 2, 3}, {0, 0, 0, 0},
                           {max_int, anchor_num, max_int, max_int});
  }

  auto unknown_dim =
      helper->Constant({1}, ONNX_NAMESPACE::TensorProto::INT64, int64_t(-1));
  auto shape_0 = helper->MakeNode(
      "Concat", {nchw[0], anchor_num_tensor, unknown_dim, nchw[2], nchw[3]});
  AddAttribute(shape_0, "axis", int64_t(0));
  auto reshaped_x = helper->MakeNode("Reshape", {x_name, shape_0->output(0)});
  auto transposed_x = helper->MakeNode("Transpose", {reshaped_x->output(0)});
  {
    std::vector<int64_t> perm({0, 1, 3, 4, 2});
    AddAttribute(transposed_x, "perm", perm);
  }

  // grid_x = np.tile(np.arange(w).reshape((1, w)), (h, 1))
  // grid_y = np.tile(np.arange(h).reshape((h, 1)), (1, w))
  auto float_value_0 =
      helper->Constant({}, GetOnnxDtype(x_info[0].dtype), float(0.0));
  auto float_value_1 =
      helper->Constant({}, GetOnnxDtype(x_info[0].dtype), float(1.0));
  auto scalar_float_w = helper->Squeeze(float_w, {});
  auto scalar_float_h = helper->Squeeze(float_h, {});
  auto grid_x_0 = helper->MakeNode(
      "Range", {float_value_0, scalar_float_w, float_value_1});  // shape is [w]
  auto grid_y_0 = helper->MakeNode(
      "Range", {float_value_0, scalar_float_h, float_value_1});  // shape is [h]
  auto grid_x_1 = helper->MakeNode(
      "Tile", {grid_x_0->output(0), nchw[2]});  // shape is [w*h]
  auto grid_y_1 = helper->MakeNode(
      "Tile", {grid_y_0->output(0), nchw[3]});  // shape is [h*w]
  auto int_value_1 =
      helper->Constant({1}, ONNX_NAMESPACE::TensorProto::INT64, float(1.0));
  auto grid_shape_x =
      helper->MakeNode("Concat", {nchw[2], nchw[3], int_value_1});
  auto grid_shape_y =
      helper->MakeNode("Concat", {nchw[3], nchw[2], int_value_1});
  AddAttribute(grid_shape_x, "axis", int64_t(0));
  AddAttribute(grid_shape_y, "axis", int64_t(0));
  auto grid_x = helper->MakeNode(
      "Reshape", {grid_x_1->output(0), grid_shape_x->output(0)});
  auto grid_y_2 = helper->MakeNode(
      "Reshape", {grid_y_1->output(0), grid_shape_y->output(0)});
  auto grid_y = helper->MakeNode("Transpose", {grid_y_2->output(0)});
  {
    std::vector<int64_t> perm({1, 0, 2});
    AddAttribute(grid_y, "perm", perm);
  }

  auto grid =
      helper->MakeNode("Concat", {grid_x->output(0), grid_y->output(0)});
  AddAttribute(grid, "axis", int64_t(2));

  // pred_box[:, :, :, :, 0] = (grid_x + sigmoid(pred_box[:, :, :, :, 0]) *
  // scale_x_y + bias_x_y) / w pred_box[:, :, :, :, 1] = (grid_y +
  // sigmoid(pred_box[:, :, :, :, 1]) * scale_x_y + bias_x_y) / h
  auto pred_box_xy =
      helper->Slice(transposed_x->output(0), {0, 1, 2, 3, 4}, {0, 0, 0, 0, 0},
                    {max_int, max_int, max_int, max_int, 2});
  auto scale_x_y =
      helper->Constant({1}, GetOnnxDtype(x_info[0].dtype), scale_x_y_);
  float bias_x_y_value = (1.0 - scale_x_y_) / 2.0;
  auto bias_x_y =
      helper->Constant({1}, GetOnnxDtype(x_info[0].dtype), bias_x_y_value);
  auto wh = helper->MakeNode("Concat", {float_w, float_h});
  AddAttribute(wh, "axis", int64_t(0));
  pred_box_xy = helper->MakeNode("Sigmoid", {pred_box_xy})->output(0);
  pred_box_xy = helper->MakeNode("Mul", {pred_box_xy, scale_x_y})->output(0);
  pred_box_xy = helper->MakeNode("Add", {pred_box_xy, bias_x_y})->output(0);
  pred_box_xy =
      helper->MakeNode("Add", {pred_box_xy, grid->output(0)})->output(0);
  pred_box_xy =
      helper->MakeNode("Div", {pred_box_xy, wh->output(0)})->output(0);

  // anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
  // anchors_s = np.array(
  //     [(an_w / input_w, an_h / input_h) for an_w, an_h in anchors])
  // anchor_w = anchors_s[:, 0:1].reshape((1, an_num, 1, 1))
  // anchor_h = anchors_s[:, 1:2].reshape((1, an_num, 1, 1))
  std::vector<int64_t> valid_anchors(anchor_num);
  valid_anchors.assign(anchors_.begin(), anchors_.begin() + anchor_num * 2);
  auto anchors = helper->Constant(GetOnnxDtype(x_info[0].dtype), valid_anchors);
  anchors = helper->Reshape(anchors, {anchor_num, 2});

  auto downsample =
      helper->Constant({1}, GetOnnxDtype(x_info[0].dtype), downsample_ratio_);
  auto ori_wh = helper->MakeNode("Mul", {wh->output(0), downsample})->output(0);
  anchors = helper->MakeNode("Div", {anchors, ori_wh})->output(0);
  // Following divide operation requires undirectional broadcast
  // It satisfies the definition of ONNX, but now sure all the inference engines
  // support this rule e.g TensorRT、OpenVINO anchor_w = anchors_s[:,
  // 0:1].reshape((1, an_num, 1, 1)) anchor_h = anchors_s[:, 1:2].reshape((1,
  // an_num, 1, 1)) pred_box[:, :, :, :, 2] = np.exp(pred_box[:, :, :, :, 2]) *
  // anchor_w pred_box[:, :, :, :, 3] = np.exp(pred_box[:, :, :, :, 3]) *
  // anchor_h
  anchors = helper->Reshape(anchors, {1, anchor_num, 1, 1, 2});
  auto pred_box_wh =
      helper->Slice(transposed_x->output(0), {0, 1, 2, 3, 4}, {0, 0, 0, 0, 2},
                    {max_int, max_int, max_int, max_int, 4});
  pred_box_wh = helper->MakeNode("Exp", {pred_box_wh})->output(0);
  pred_box_wh = helper->MakeNode("Mul", {pred_box_wh, anchors})->output(0);

  // if iou_aware:
  //     pred_conf = sigmoid(x[:, :, :, :, 4:5])**(
  //         1 - iou_aware_factor) * sigmoid(ioup)**iou_aware_factor
  // else:
  //     pred_conf = sigmoid(x[:, :, :, :, 4:5])
  auto confidence =
      helper->Slice(transposed_x->output(0), {0, 1, 2, 3, 4}, {0, 0, 0, 0, 4},
                    {max_int, max_int, max_int, max_int, 5});
  std::string pred_conf = helper->MakeNode("Sigmoid", {confidence})->output(0);
  if (iou_aware_) {
    auto ioup = helper->Slice(x_info[0].name, {0, 1, 2, 3}, {0, 0, 0, 0},
                              {max_int, anchor_num, max_int, max_int});
    ioup = helper->Unsqueeze(ioup, {4});
    ioup = helper->MakeNode("Sigmoid", {ioup})->output(0);
    float power_value_0 = 1 - iou_aware_factor_;
    auto power_0 =
        helper->Constant({1}, GetOnnxDtype(x_info[0].dtype), power_value_0);
    auto power_1 =
        helper->Constant({1}, GetOnnxDtype(x_info[0].dtype), iou_aware_factor_);
    ioup = helper->MakeNode("Pow", {ioup, power_1})->output(0);
    pred_conf = helper->MakeNode("Pow", {pred_conf, power_0})->output(0);
    pred_conf = helper->MakeNode("Mul", {pred_conf, ioup})->output(0);
  }

  // pred_conf[pred_conf < conf_thresh] = 0.
  // pred_score = sigmoid(x[:, :, :, :, 5:]) * pred_conf
  // pred_box = pred_box * (pred_conf > 0.).astype('float32')
  auto value_2 =
      helper->Constant({1}, GetOnnxDtype(x_info[0].dtype), float(2.0));
  auto center = helper->MakeNode("Div", {pred_box_wh, value_2})->output(0);
  auto min_xy = helper->MakeNode("Sub", {pred_box_xy, center})->output(0);
  auto max_xy = helper->MakeNode("Add", {pred_box_xy, center})->output(0);

  auto conf_thresh =
      helper->Constant({1}, GetOnnxDtype(x_info[0].dtype), conf_thresh_);
  auto filter =
      helper->MakeNode("Greater", {pred_conf, conf_thresh})->output(0);
  filter = helper->AutoCast(filter, P2ODataType::BOOL, x_info[0].dtype);
  pred_conf = helper->MakeNode("Mul", {pred_conf, filter})->output(0);
  auto pred_score =
      helper->Slice(transposed_x->output(0), {0, 1, 2, 3, 4}, {0, 0, 0, 0, 5},
                    {max_int, max_int, max_int, max_int, max_int});
  pred_score = helper->MakeNode("Sigmoid", {pred_score})->output(0);
  pred_score = helper->MakeNode("Mul", {pred_score, pred_conf})->output(0);
  auto pred_box = helper->Concat({min_xy, max_xy}, 4);
  pred_box = helper->MakeNode("Mul", {pred_box, filter})->output(0);

  auto value_neg_1 =
      helper->Constant({1}, ONNX_NAMESPACE::TensorProto::INT64, int64_t(-1));
  auto value_4 =
      helper->Constant({1}, ONNX_NAMESPACE::TensorProto::INT64, int64_t(4));
  auto new_shape = helper->Concat({nchw[0], value_neg_1, value_4}, 0);
  pred_box = helper->MakeNode("Reshape", {pred_box, new_shape})->output(0);

  auto float_img_size = helper->AutoCast(
      im_size_info[0].name, im_size_info[0].dtype, x_info[0].dtype);
  float_img_size = helper->Unsqueeze(float_img_size, {1});
  auto split_im_hw = helper->Split(float_img_size, {1, 1}, 2);
  auto im_whwh = helper->Concat(
      {split_im_hw[1], split_im_hw[0], split_im_hw[1], split_im_hw[0]}, 2);

  if (!clip_bbox_) {
    auto out = helper->MakeNode("Mul", {pred_box, im_whwh})->output(0);
    helper->AutoCast(out, boxes_info[0].name, x_info[0].dtype,
                     boxes_info[0].dtype);
  } else {
    pred_box = helper->MakeNode("Mul", {pred_box, im_whwh})->output(0);
    auto im_wh = helper->Concat({split_im_hw[1], split_im_hw[0]}, 2);
    im_wh = helper->MakeNode("Sub", {im_wh, float_value_1})->output(0);
    auto pred_box_xymin_xymax = helper->Split(pred_box, {2, 2}, 2);
    pred_box_xymin_xymax[0] =
        helper->MakeNode("Relu", {pred_box_xymin_xymax[0]})->output(0);
    pred_box_xymin_xymax[1] =
        helper->MakeNode("Min", {pred_box_xymin_xymax[1], im_wh})->output(0);
    auto out = helper->Concat(pred_box_xymin_xymax, 2);
    helper->AutoCast(out, boxes_info[0].name, x_info[0].dtype,
                     boxes_info[0].dtype);
  }

  auto class_num =
      helper->Constant({1}, ONNX_NAMESPACE::TensorProto::INT64, class_num_);
  auto score_out_shape =
      helper->Concat({nchw[0], value_neg_1, class_num}, int64_t(0));
  auto score_out =
      helper->MakeNode("Reshape", {pred_score, score_out_shape})->output(0);
  helper->AutoCast(score_out, scores_info[0].name, x_info[0].dtype,
                   scores_info[0].dtype);
}
}  // namespace paddle2onnx
