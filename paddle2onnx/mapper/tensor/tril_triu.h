
#pragma once
#include <string>
#include <vector>

#include "paddle2onnx/mapper/mapper.h"

namespace paddle2onnx {

class TrilTriuMapper : public Mapper {
 public:
  TrilTriuMapper(const PaddleParser& p, OnnxHelper* helper, int64_t block_id,
             int64_t op_id)
      : Mapper(p, helper, block_id, op_id) {
        if (HasAttr("diagonal")) {
            GetAttr("diagonal", &diagonal_);
        }
        if (HasAttr("name")) {
            GetAttr("name", &triu_name_);
        }
        if (HasAttr("lower")){
            GetAttr("lower", &lower_);
        }
      }
  
  int32_t GetMinOpset(bool verbose = false) override;
  void Opset14() override;
private:
  int64_t diagonal_ = 0;
  bool lower_ = true;
  std::string triu_name_ = "None";
};
}