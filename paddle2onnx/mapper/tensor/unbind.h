#pragma once
#include <string>
#include <vector>

#include "paddle2onnx/mapper/mapper.h"

namespace paddle2onnx {

class UnbindMapper : public Mapper {
 public:
  UnbindMapper(const PaddleParser& p, OnnxHelper* helper, int64_t block_id,
             int64_t op_id)
      : Mapper(p, helper, block_id, op_id) {
        GetAttr("axis", &axis_);
      }
  void Opset7();
  int64_t axis_;
};

}  // namespace paddle2onnx