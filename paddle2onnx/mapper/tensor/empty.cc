#include "paddle2onnx/mapper/tensor/empty.h"

namespace paddle2onnx
{
  REGISTER_MAPPER(empty, EmptyMapper)

  int32_t EmptyMapper::GetMinOpset(bool verbose)
  {
    constexpr int op_version = 11;
    Logger(verbose, op_version) << RequireOpset(op_version) << std::endl;
    return op_version;
  }

  void EmptyMapper::Opset11()
  {
    std::vector<TensorInfo> out_info = GetOutput("Out");
    // shape tensor/tensor list/tuple
    bool shape_is_tensor = HasInput("ShapeTensor");
    bool shape_is_tensor_list = HasInput("ShapeTensorList");
    bool shape_is_other_types = !(shape_is_tensor || shape_is_tensor_list);
    // Paddle-model output dtype to onnx-model dtype
    ONNX_NAMESPACE::TensorProto_DataType onnx_dtype = GetOnnxDtype(out_info[0].dtype);
    // Fill with 0
    float value = 0;
    // a) If shape is list or tuple (that means it's not a variable), we can use constant op directly
    if (shape_is_other_types)
    {
      std::vector<int64_t> shape;
      GetAttr("shape", &shape);
      helper_->Constant(out_info[0].name, shape, onnx_dtype, value);// The acceptable (shape) argument must be vector<int64_t> 
      return;
    }
    // b) If shape is tensor (variable), we should cast them to INT64.
    // c) If shape is tensorlist (variable), we should cast them to INT64 and concat all tensors (the ConcatIndices function has implemented the cast).
    std::string shape_name;
    if (shape_is_tensor)
    {
      std::vector<TensorInfo> shape_info = GetInput("ShapeTensor");
      shape_name = helper_->AutoCast(shape_info[0].name, shape_info[0].dtype, P2ODataType::INT64);
    }
    else //tensor list
    {
      std::vector<TensorInfo> shape_info = GetInput("ShapeTensorList");
      shape_name = helper_->ConcatIndices(shape_info);
    }
    auto node = helper_->MakeNode("ConstantOfShape", {shape_name}, {out_info[0].name});

    // The attribute [value] of ConstantOfShape op, a one-element tensor, is the value filled in output.
    auto attr = node->add_attribute();
    attr->set_name("value");                                // attribute name
    attr->set_type(ONNX_NAMESPACE::AttributeProto::TENSOR); // attribute dtype
    auto tensor = attr->mutable_t();
    tensor->set_name(out_info[0].name);
    tensor->set_data_type(onnx_dtype);                      // onnx dytpe, not a paddle dtype
    tensor->add_dims(1);                                    // one dimension tensor with one element
    if (onnx_dtype == ONNX_NAMESPACE::TensorProto::INT32)
    {
      std::vector<int32_t> data(1);
      data[0] = static_cast<int32_t>(value);
      const char *ptr = reinterpret_cast<const char *>(data.data());
      tensor->set_raw_data(std::string(ptr, sizeof(int32_t)));
    }
    else if (onnx_dtype == ONNX_NAMESPACE::TensorProto::INT64)
    {
      std::vector<int64_t> data(1);
      data[0] = static_cast<int64_t>(value);
      const char *ptr = reinterpret_cast<const char *>(data.data());
      tensor->set_raw_data(std::string(ptr, sizeof(int64_t)));
    }
    else if (onnx_dtype == ONNX_NAMESPACE::TensorProto::FLOAT)
    {
      std::vector<float> data(1, value); // float do not need to be converted.
      const char *ptr = reinterpret_cast<const char *>(data.data());
      tensor->set_raw_data(std::string(ptr, sizeof(float)));
    }
    else if (onnx_dtype == ONNX_NAMESPACE::TensorProto::DOUBLE)
    {
      std::vector<double> data(1);
      data[0] = static_cast<double>(value);
      const char *ptr = reinterpret_cast<const char *>(data.data());
      tensor->set_raw_data(std::string(ptr, sizeof(double)));
    }
    else if (onnx_dtype == ONNX_NAMESPACE::TensorProto::BOOL)
    {
      // std::vector<bool> is a specialized container class that stores data not as a byte per Boolean value, but as a bit compression.
      // This makes the direct use of std::vector<bool> potentially problematic
      bool *data = new bool[1];
      data[0] = static_cast<bool>(value);
      tensor->set_raw_data(std::string((const char *)(data), sizeof(bool)));
      delete[] data;
    }
  }
} // namespace paddle2onnx