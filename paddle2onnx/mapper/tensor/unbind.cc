#include "paddle2onnx/mapper/tensor/unbind.h"

namespace paddle2onnx {
REGISTER_MAPPER(unbind, UnbindMapper)

void UnbindMapper::Opset7() {
  auto input_info = GetInput("X");
  auto output_info = GetOutput("Out");
  
  std::vector<std::string> output_names(output_info.size());
  for (size_t i = 0; i < output_info.size(); ++i) {
    output_names[i] = output_info[i].name;
  }


  int64_t split_axis = axis_;
  if (split_axis < 0) {
    split_axis += input_info[0].Rank();
  }
  
  std::vector<int64_t> split_sizes = std::vector<int64_t>(input_info[0].shape[split_axis],1);
  helper_->Split(input_info[0].name, output_names, split_sizes, split_axis);

  for (size_t i = 0; i < output_info.size(); ++i) {
    std::vector<int64_t> axes{split_axis}; 
    helper_->Squeeze(output_names[i], output_names[i], axes);
  }
}
}