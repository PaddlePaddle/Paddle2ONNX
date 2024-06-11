
#pragma once
#include <string>
#include <vector>

#include "paddle2onnx/mapper/mapper.h"

namespace paddle2onnx {

class EmptyMapper : public Mapper {
 public:
  EmptyMapper(const PaddleParser& p, OnnxHelper* helper, int64_t block_id,
             int64_t op_id)
      : Mapper(p, helper, block_id, op_id) {
        GetAttr("dtype", &output_dtype_);
      }
  
  int32_t GetMinOpset(bool verbose = false) override;
  void Opset11() override;
private:
  int64_t output_dtype_;
};
}