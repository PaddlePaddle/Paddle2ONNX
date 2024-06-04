#include "paddle2onnx/mapper/tensor/tril_triu.h"

namespace paddle2onnx {
REGISTER_MAPPER(tril_triu, TrilTriuMapper)

int32_t TrilTriuMapper::GetMinOpset(bool verbose) {
  constexpr int op_version = 14;
  Logger(verbose, op_version) << RequireOpset(op_version) << std::endl;
  return op_version;
}


void TrilTriuMapper::Opset14() {
  auto x_info = GetInput("X");
  auto out_info = GetOutput("Out");


  std::vector<int64_t> diagonal_vec{diagonal_};
  
  std::string diagonal_node_name = helper_->Constant(ONNX_NAMESPACE::TensorProto::INT64, diagonal_vec);
  auto output_node = helper_->MakeNode("Trilu", {x_info[0].name, diagonal_node_name}, {out_info[0].name});
  int64_t upper = !lower_;
  AddAttribute(output_node, "upper", upper);
}
}  // namespace paddle2onnx