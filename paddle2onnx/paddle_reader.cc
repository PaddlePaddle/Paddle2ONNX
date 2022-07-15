#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include "paddle2onnx/converter.h"
#include "paddle2onnx/mapper/exporter.h"

namespace paddle2onnx {

void GetOpAttr(const paddle2onnx::framework::proto::OpDesc& op,
               const std::string& name, int64_t* res) {
  bool found = false;
  for (auto i = 0; i < op.attrs_size(); ++i) {
    if (op.attrs(i).name() == name) {
      found = true;
      Assert(op.attrs(i).has_i() || op.attrs(i).has_l(),
             "Cannot find int32/int64 data from attr: " + name +
                 " in op:" + op.type());
      if (op.attrs(i).has_i()) {
        *res = (int64_t)(op.attrs(i).i());
      } else {
        *res = op.attrs(i).l();
      }
      break;
    }
  }
  Assert(found, "Cannot found attribute " + name + " in op: " + op.type());
}

PADDLE2ONNX_DECL PaddleReader::PaddleReader(const char* model_buffer,
                                            int buffer_size) {
  auto prog = std::make_shared<paddle2onnx::framework::proto::ProgramDesc>();
  std::string content(model_buffer, model_buffer + buffer_size);
  Assert(prog->ParseFromString(content), "Failed to parse PaddlePaddle model.");

  num_inputs = 0;
  num_outputs = 0;
  for (auto i = 0; i < prog->blocks(0).ops_size(); ++i) {
    if (prog->blocks(0).ops(i).type() == "fetch") {
      std::string name = prog->blocks(0).ops(i).inputs(0).arguments(0);
      int64_t order = -1;
      GetOpAttr(prog->blocks(0).ops(i), "col", &order);
      Assert(order > 0, "Find invalid order less than 0 in fetch op.");
      Assert(order < 100,
             "Find invalid order which bigger than 99 in fetch op.");
      num_outputs += 1;
      memcpy(output_names[order], name.c_str(), name.size());
    } else if (prog->blocks(0).ops(i).type() == "feed") {
      std::string name = prog->blocks(0).ops(i).outputs(0).arguments(0);
      int64_t order = -1;
      GetOpAttr(prog->blocks(0).ops(i), "col", &order);
      Assert(order > 0, "Find invalid order less than 0 in feed op.");
      Assert(order < 100,
             "Find invalid order which bigger than 99 in feed op.");
      num_inputs += 1;
      memcpy(input_names[order], name.c_str(), name.size());
    }
    has_nms = !(prog->blocks(0).ops(i).type().find("nms") == std::string::npos);
  }
}

PADDLE2ONNX_DECL int PaddleReader::NumInputs() const { return num_inputs; }

PADDLE2ONNX_DECL int PaddleReader::NumOutputs() const { return num_outputs; }

PADDLE2ONNX_DECL int PaddleReader::GetInputIndex(const char* name) const {
  for (int i = 0; i < num_inputs; ++i) {
    if (strcmp(name, input_names[i]) == 0) {
      return i;
    }
  }
  return -1;
}

PADDLE2ONNX_DECL int PaddleReader::GetOutputIndex(const char* name) const {
  for (int i = 0; i < num_outputs; ++i) {
    if (strcmp(name, output_names[i]) == 0) {
      return i;
    }
  }
  return -1;
}

}  // namespace paddle2onnx
