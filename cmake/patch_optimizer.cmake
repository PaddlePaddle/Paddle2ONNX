# Patch onnx-optimizer for compatibility with ONNX 1.20.1
# Dimension constructor is now explicit, so fix implicit conversions

file(READ "onnxoptimizer/passes/fuse_add_bias_into_conv.h" content)
string(REPLACE
  "std::vector<Dimension> s = {1};"
  "std::vector<Dimension> s = {Dimension(1)};"
  content "${content}")
file(WRITE "onnxoptimizer/passes/fuse_add_bias_into_conv.h" "${content}")
