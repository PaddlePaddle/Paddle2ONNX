#include "paddle2onnx/mapper/tensor/triu.h"

namespace paddle2onnx {
REGISTER_MAPPER(tril_triu, TriuMapper)

int32_t TriuMapper::GetMinOpset(bool verbose) {
  constexpr int op_version = 14;
  Logger(verbose, op_version) << RequireOpset(op_version) << std::endl;
  return op_version;
}


void TriuMapper::Opset14() {
  auto x_info = GetInput("X");
  auto out_info = GetOutput("Out");

  std::vector<int64_t> diagonal_vec{diagonal_};
  std::string diagonal_node_name = helper_->Constant(ONNX_NAMESPACE::TensorProto::INT64, diagonal_vec);

  auto output_node = helper_->MakeNode("Trilu", {x_info[0].name, diagonal_node_name}, {out_info[0].name});
 // AddAttribute(output_node, "name", triu_name_);
}
}  // namespace paddle2onnx