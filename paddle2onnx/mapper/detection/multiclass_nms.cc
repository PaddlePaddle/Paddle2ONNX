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

#include "paddle2onnx/mapper/detection/multiclass_nms.h"

namespace paddle2onnx {

REGISTER_MAPPER(multiclass_nms3, NMSMapper);

int32_t NMSMapper::GetMinOpset(bool verbose) {
  std::vector<TensorInfo> boxes_info =
      parser_->GetOpInput(block_idx_, op_idx_, "BBoxes");
  std::vector<TensorInfo> score_info =
      parser_->GetOpInput(block_idx_, op_idx_, "Scores");
  if (score_info[0].Rank() != 3) {
    if (verbose) {
      std::cerr << "Paddle2ONNX: Lod Tensor input is not supported in "
                   "multiclass_nms3 operator, which means the shape of "
                   "input(scores) is [M, C] now, but it desire [N, C, M]."
                << std::endl;
    }
    return -1;
  }
  if (boxes_info[0].Rank() != 3) {
    if (verbose) {
      std::cerr << "Paddle2ONNX: Only support input boxes as 3-D Tensor in  "
                   "multiclass_nms3 operator, but now it's rank is "
                << boxes_info[0].Rank() << "." << std::endl;
    }
    return -1;
  }
  if (boxes_info[0].shape[1] < 0 || boxes_info[0].shape[2] < 0) {
    if (verbose) {
      std::cerr << "Paddle2ONNX: The 2nd and 3rd dimension of input bboxes "
                   "tensor of multiclass_nms should be fixed, but now the "
                   "shape is ["
                << boxes_info[0].shape[0] << ", " << boxes_info[0].shape[1]
                << ", " << boxes_info[0].shape[2] << "]." << std::endl;
    }
    return -1;
  }
  if (score_info[0].shape[1] < 0 || score_info[0].shape[2] < 0) {
    if (verbose) {
      std::cerr << "Paddle2ONNX: The 2nd and 3rd dimension of input scores "
                   "tensor of multiclass_nms should be fixed, but now the "
                   "shape is ["
                << score_info[0].shape[0] << ", " << score_info[0].shape[1]
                << ", " << score_info[0].shape[2] << "]." << std::endl;
    }
    return -1;
  }
  return 10;
}

void NMSMapper::KeepTopK(OnnxHelper* helper,
                         const std::string& selected_indices) {
  auto boxes_info = parser_->GetOpInput(block_idx_, op_idx_, "BBoxes");
  auto score_info = parser_->GetOpInput(block_idx_, op_idx_, "Scores");
  auto out_info = parser_->GetOpOutput(block_idx_, op_idx_, "Out");
  auto index_info = parser_->GetOpOutput(block_idx_, op_idx_, "Index");
  auto num_rois_info = parser_->GetOpOutput(block_idx_, op_idx_, "NmsRoisNum");
  auto value_0 =
      helper->Constant({1}, ONNX_NAMESPACE::TensorProto::INT64, int64_t(0));
  auto value_1 =
      helper->Constant({1}, ONNX_NAMESPACE::TensorProto::INT64, int64_t(1));
  auto value_2 =
      helper->Constant({1}, ONNX_NAMESPACE::TensorProto::INT64, int64_t(2));
  auto value_neg_1 =
      helper->Constant({1}, ONNX_NAMESPACE::TensorProto::INT64, int64_t(-1));

  auto class_id = helper->MakeNode("Gather", {selected_indices, value_1});
  AddAttribute(class_id, "axis", int64_t(1));

  auto box_id = helper->MakeNode("Gather", {selected_indices, value_2});
  AddAttribute(box_id, "axis", int64_t(1));

  auto filtered_class_id = class_id->output(0);
  auto filtered_box_id = box_id->output(0);
  if (background_label_ >= 0) {
    auto filter_indices = MapperHelper::Get()->GenName("nms.filter_background");
    auto squeezed_class_id =
        helper->Squeeze(class_id->output(0), std::vector<int64_t>(1, 1));
    if (background_label_ > 0) {
      auto background = helper->Constant(
          {1}, ONNX_NAMESPACE::TensorProto::INT64, background_label_);
      auto diff = helper->MakeNode("Sub", {squeezed_class_id, background});
      helper->MakeNode("NonZero", {diff->output(0)}, {filter_indices});
    } else if (background_label_ == 0) {
      helper->MakeNode("NonZero", {squeezed_class_id}, {filter_indices});
    }
    auto new_class_id =
        helper->MakeNode("Gather", {filtered_class_id, filter_indices});
    AddAttribute(new_class_id, "axis", int64_t(0));
    auto new_box_id =
        helper->MakeNode("Gather", {box_id->output(0), filter_indices});
    AddAttribute(new_box_id, "axis", int64_t(0));
    filtered_class_id = new_class_id->output(0);
    filtered_box_id = new_box_id->output(0);
  }

  // Here is a little complicated
  // Since we need to gather all the scores for the final boxes to filter the
  // top-k boxes Now we have the follow inputs
  //    - scores: [N, C, M] N means batch size(but now it will be regarded as
  //    1); C means number of classes; M means number of boxes for each classes
  //    - selected_indices: [num_selected_indices, 3], and 3 means [batch,
  //    class_id, box_id]. We will use this inputs to gather score
  // So now we will first flatten `scores` to shape of [1 * C * M], then we
  // gather scores by each elements in `selected_indices` The index need be
  // calculated as
  //    `gather_index = class_id * M + box_id`
  auto flatten_score = helper->Flatten(score_info[0].name);
  auto num_boxes_each_class = helper->Constant(
      {1}, ONNX_NAMESPACE::TensorProto::INT64, score_info[0].shape[2]);
  auto gather_indices_0 =
      helper->MakeNode("Mul", {filtered_class_id, num_boxes_each_class});
  auto gather_indices_1 =
      helper->MakeNode("Add", {gather_indices_0->output(0), filtered_box_id});
  auto gather_indices = helper->Flatten(gather_indices_1->output(0));
  auto gathered_scores =
      helper->MakeNode("Gather", {flatten_score, gather_indices});
  AddAttribute(gathered_scores, "axis", int64_t(0));

  // Now we will perform keep_top_k process
  // First we need to check if the number of remaining boxes is greater than
  // keep_top_k Otherwise, we will downgrade the keep_top_k to number of
  // remaining boxes
  auto final_classes = filtered_class_id;
  auto final_boxes_id = filtered_box_id;
  auto final_scores = gathered_scores->output(0);
  if (keep_top_k_ > 0) {
    // get proper topk
    auto shape_of_scores = helper->MakeNode("Shape", {final_scores});
    auto num_of_boxes =
        helper->Slice(shape_of_scores->output(0), std::vector<int64_t>(1, 0),
                      std::vector<int64_t>(1, 0), std::vector<int64_t>(1, 1));
    auto top_k =
        helper->Constant({1}, ONNX_NAMESPACE::TensorProto::INT64, keep_top_k_);
    auto ensemble_value = helper->MakeNode("Concat", {num_of_boxes, top_k});
    AddAttribute(ensemble_value, "axis", int64_t(0));
    auto new_top_k = helper->MakeNode("ReduceMin", {ensemble_value->output(0)});
    AddAttribute(new_top_k, "axes", std::vector<int64_t>(1, 0));
    AddAttribute(new_top_k, "keepdims", int64_t(1));

    // the output is topk_scores, topk_score_indices
    auto topk_node =
        helper->MakeNode("TopK", {final_scores, new_top_k->output(0)}, 2);
    auto topk_scores =
        helper->MakeNode("Gather", {final_scores, topk_node->output(1)});
    AddAttribute(topk_scores, "axis", int64_t(0));
    auto topk_classes =
        helper->MakeNode("Gather", {filtered_class_id, topk_node->output(1)});
    AddAttribute(topk_classes, "axis", int64_t(1));
    auto topk_boxes_id =
        helper->MakeNode("Gather", {filtered_box_id, topk_node->output(1)});
    AddAttribute(topk_boxes_id, "axis", int64_t(1));

    final_boxes_id = topk_boxes_id->output(0);
    final_scores = topk_scores->output(0);
    final_classes = topk_classes->output(0);
  }

  auto flatten_boxes_id = helper->Flatten({final_boxes_id});
  auto gathered_selected_boxes =
      helper->MakeNode("Gather", {boxes_info[0].name, flatten_boxes_id});
  AddAttribute(gathered_selected_boxes, "axis", int64_t(1));

  auto float_classes = helper->MakeNode("Cast", {final_classes});
  AddAttribute(float_classes, "to", ONNX_NAMESPACE::TensorProto::FLOAT);

  std::vector<int64_t> shape{1, -1, 1};
  auto unsqueezed_scores = helper->Reshape({final_scores}, shape);

  auto box_result =
      helper->MakeNode("Concat", {float_classes->output(0), unsqueezed_scores,
                                  gathered_selected_boxes->output(0)});
  AddAttribute(box_result, "axis", int64_t(2));
  helper->Squeeze({box_result->output(0)}, {out_info[0].name},
                  std::vector<int64_t>(1, 0));

  // other outputs, we don't use sometimes
  // there's lots of Cast in exporting
  // TODO(jiangjiajun) A pass to eleminate all the useless Cast is needed
  auto reshaped_index_result =
      helper->Reshape({flatten_boxes_id}, {int64_t(-1), int64_t(1)});
  auto index_result =
      helper->MakeNode("Cast", {reshaped_index_result}, {index_info[0].name});
  AddAttribute(index_result, "to", GetOnnxDtype(index_info[0].dtype));

  auto out_box_shape = helper->MakeNode("Shape", {box_result->output(0)});
  auto num_rois_result =
      helper->Slice({out_box_shape->output(0)}, std::vector<int64_t>(1, 0),
                    std::vector<int64_t>(1, 0), std::vector<int64_t>(1, 1));
  auto int32_num_rois_result =
      helper->MakeNode("Cast", {num_rois_result}, {num_rois_info[0].name});
  AddAttribute(int32_num_rois_result, "to",
               GetOnnxDtype(num_rois_info[0].dtype));
}

void NMSMapper::Opset10(OnnxHelper* helper) {
  std::vector<TensorInfo> boxes_info =
      parser_->GetOpInput(block_idx_, op_idx_, "BBoxes");
  std::vector<TensorInfo> score_info =
      parser_->GetOpInput(block_idx_, op_idx_, "Scores");
  if (boxes_info[0].shape[0] != 1) {
    std::cerr << "[WARN] Due to the operator multiclass_nms, the exported ONNX "
                 "model will only supports inference with input batch_size == "
                 "1."
              << std::endl;
  }
  int64_t num_classes = score_info[0].shape[1];
  auto score_threshold = helper->Constant(
      {1}, ONNX_NAMESPACE::TensorProto::FLOAT, score_threshold_);
  auto nms_threshold =
      helper->Constant({1}, ONNX_NAMESPACE::TensorProto::FLOAT, nms_threshold_);
  auto nms_top_k =
      helper->Constant({1}, ONNX_NAMESPACE::TensorProto::INT64, nms_top_k_);

  auto selected_box_index = MapperHelper::Get()->GenName("nms.selected_index");
  if (normalized_) {
    helper->MakeNode("NonMaxSuppression",
                     {boxes_info[0].name, score_info[0].name, nms_top_k,
                      nms_threshold, score_threshold},
                     {selected_box_index});
  } else {
    auto value_1 =
        helper->Constant({1}, GetOnnxDtype(boxes_info[0].dtype), float(1.0));
    auto split_boxes = helper->MakeNode("Split", {boxes_info[0].name}, 4);
    AddAttribute(split_boxes, "axis", int64_t(2));
    AddAttribute(split_boxes, "split", std::vector<int64_t>(4, 1));
    auto xmax = helper->MakeNode("Add", {split_boxes->output(2), value_1});
    auto ymax = helper->MakeNode("Add", {split_boxes->output(3), value_1});
    auto new_boxes = helper->MakeNode(
        "Concat", {split_boxes->output(0), split_boxes->output(1),
                   xmax->output(0), ymax->output(0)});
    AddAttribute(new_boxes, "axis", int64_t(2));
    helper->MakeNode("NonMaxSuppression",
                     {new_boxes->output(0), score_info[0].name, nms_top_k,
                      nms_threshold, score_threshold},
                     {selected_box_index});
  }
  KeepTopK(helper, selected_box_index);
}
}  // namespace paddle2onnx
