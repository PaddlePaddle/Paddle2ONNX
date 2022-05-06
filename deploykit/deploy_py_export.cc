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

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <type_traits>

#ifdef ENABLE_ORT_BACKEND
#include "deploykit/backends/ort/ort_backend.h"
#endif

#ifdef ENABLE_TRT_BACKEND
#include "deploykit/backends/tensorrt/trt_backend.h"
#endif

namespace deploykit {

pybind11::dtype PaddleDataTypeToNumpyDataType(const int& paddle_dtype) {
  pybind11::dtype dt;
  switch (paddle_dtype) {
    case PaddleDataType::INT32:
      dt = pybind11::dtype::of<int32_t>();
      break;
    case PaddleDataType::INT64:
      dt = pybind11::dtype::of<int64_t>();
      break;
    case PaddleDataType::FP32:
      dt = pybind11::dtype::of<float>();
      break;
    case PaddleDataType::FP64:
      dt = pybind11::dtype::of<double>();
      break;
    default:
      Assert(false, "Cannot handle paddle data type: " +
                        std::to_string(paddle_dtype) +
                        " while calling PaddleDataTypeToNumpyDataType.");
  }
  return dt;
}

int NumpyDataTypeToPaddleDataType(const pybind11::dtype& np_dtype) {
  if (np_dtype.is(pybind11::dtype::of<int32_t>())) {
    return PaddleDataType::INT32;
  } else if (np_dtype.is(pybind11::dtype::of<int64_t>())) {
    return PaddleDataType::INT64;
  } else if (np_dtype.is(pybind11::dtype::of<float>())) {
    return PaddleDataType::FP32;
  } else if (np_dtype.is(pybind11::dtype::of<double>())) {
    return PaddleDataType::FP64;
  }
  Assert(false,
         "NumpyDataTypeToPaddleDataType() only support "
         "int32/int64/float32/float64 now.");
  return PaddleDataType::FP32;
}

template <typename T>
int CTypeToPaddleDataType() {
  if (std::is_same<T, int32_t>::value) {
    return PaddleDataType::INT32;
  } else if (std::is_same<T, int64_t>::value) {
    return PaddleDataType::INT64;
  } else if (std::is_same<T, float>::value) {
    return PaddleDataType::FP32;
  } else if (std::is_same<T, double>::value) {
    return PaddleDataType::FP64;
  }
  Assert(false,
         "CTypeToPaddleDataType only support int32/int64/float32/float64 now.");
  return PaddleDataType::FP32;
}

template <typename T>
std::vector<pybind11::array> PyBackendInfer(
    T& self, const std::vector<std::string>& names,
    std::vector<pybind11::array>& data) {
  std::vector<DataBlob> inputs(data.size());
  for (size_t i = 0; i < data.size(); ++i) {
    inputs[i].dtype = NumpyDataTypeToPaddleDataType(data[i].dtype());
    inputs[i].shape.insert(inputs[i].shape.begin(), data[i].shape(),
                           data[i].shape() + data[i].ndim());
    inputs[i].py_array_t = data[i].mutable_data();
    inputs[i].data.resize(data[i].nbytes());
    memcpy(inputs[i].data.data(), data[i].mutable_data(), data[i].nbytes());
    inputs[i].name = names[i];
  }

  std::vector<DataBlob> outputs(self.NumOutputs());
  self.Infer(inputs, &outputs);

  std::vector<pybind11::array> results;
  results.reserve(outputs.size());
  for (size_t i = 0; i < outputs.size(); ++i) {
    auto numpy_dtype = PaddleDataTypeToNumpyDataType(outputs[i].dtype);
    results.emplace_back(pybind11::array(numpy_dtype, outputs[i].shape));
    memcpy(results[i].mutable_data(), outputs[i].data.data(),
           outputs[i].Numel() * PaddleDataTypeSize(outputs[i].dtype));
  }
  return results;
}

PYBIND11_MODULE(deploykit_cpp2py_export, m) {
  m.doc() = "Paddle Deployment Toolkit.";

#ifdef ENABLE_ORT_BACKEND
  // onnxruntime backend
  pybind11::class_<OrtBackendOption>(m, "OrtBackendOption")
      .def(pybind11::init())
      .def_readwrite("graph_optimization_level",
                     &OrtBackendOption::graph_optimization_level)
      .def_readwrite("inter_op_num_threads",
                     &OrtBackendOption::inter_op_num_threads)
      .def_readwrite("intra_op_num_threads",
                     &OrtBackendOption::intra_op_num_threads)
      .def_readwrite("use_gpu", &OrtBackendOption::use_gpu)
      .def_readwrite("gpu_id", &OrtBackendOption::gpu_id)
      .def_readwrite("execution_mode", &OrtBackendOption::execution_mode);
  pybind11::class_<OrtBackend>(m, "OrtBackend")
      .def(pybind11::init())
      .def("load_paddle",
           [](OrtBackend& self, const std::string& model_file,
              const std::string& params_file, const OrtBackendOption& option,
              bool debug) {
             return self.InitFromPaddle(model_file, params_file, option, debug);
           })
      .def("load_onnx",
           [](OrtBackend& self, const std::string& file_name,
              const OrtBackendOption& option) {
             return self.InitFromOnnx(file_name, option);
           })
      .def("infer", &PyBackendInfer<OrtBackend>);
#endif

#ifdef ENABLE_TRT_BACKEND
  // tensorrt backend
  pybind11::class_<TrtBackendOption>(m, "TrtBackendOption")
      .def(pybind11::init())
      .def_readwrite("gpu_id", &TrtBackendOption::gpu_id)
      .def_readwrite("fixed_shape", &TrtBackendOption::fixed_shape)
      .def_readwrite("serialize_file", &TrtBackendOption::serialize_file)
      .def_readwrite("max_batch_size", &TrtBackendOption::max_batch_size)
      .def_readwrite("max_shape", &TrtBackendOption::max_shape)
      .def_readwrite("opt_shape", &TrtBackendOption::opt_shape)
      .def_readwrite("min_shape", &TrtBackendOption::min_shape)
      .def_readwrite("enable_fp16", &TrtBackendOption::enable_fp16)
      .def_readwrite("enable_int8", &TrtBackendOption::enable_int8);
  pybind11::class_<TrtBackend>(m, "TrtBackend")
      .def(pybind11::init())
      .def("load_paddle",
           [](TrtBackend& self, const std::string& model_file,
              const std::string& params_file, const TrtBackendOption& option,
              bool debug) {
             return self.InitFromPaddle(model_file, params_file, option, debug);
           })
      .def("load_onnx",
           [](TrtBackend& self, const std::string& file_name,
              const TrtBackendOption& option) {
             return self.InitFromOnnx(file_name, option);
           })
      .def("infer", &PyBackendInfer<TrtBackend>);
#endif
}

}  // namespace deploykit
