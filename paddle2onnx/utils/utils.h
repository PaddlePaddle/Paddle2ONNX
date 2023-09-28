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

#pragma once
#include <stdlib.h>

#include <iostream>
#include <sstream>
#include <string>
#include <cstring>

namespace paddle2onnx {

inline void Assert(bool condition, const std::string& message) {
  if (!condition) {
    fprintf(stderr, "[ERROR] %s\n", message.c_str());
    std::abort();
  }
}

inline const std::string RequireOpset(const int32_t& opset_version) {
  return "Requires the minimal opset version of " +
         std::to_string(opset_version) + ".";
}

// from https://blog.csdn.net/q2519008/article/details/129264884
inline uint16_t FP32ToFP16(float fp32_num)
{
  uint32_t temp_data;
  memcpy(&temp_data,&fp32_num,sizeof(float));
  uint16_t t = ((temp_data & 0x007fffff) >> 13) | ((temp_data & 0x80000000) >> 16) | (((temp_data & 0x7f800000) >> 13) - (112 << 10));           
  if(temp_data & 0x1000) {
    t++;   
  }                
  uint16_t fp16 = *(uint16_t*)(&t);     
  return fp16;
}

class P2OLogger {
 public:
  P2OLogger() {
    line_ = "";
    prefix_ = "[Paddle2ONNX]";
    verbose_ = true;
  }
  explicit P2OLogger(bool verbose,
                     const std::string& prefix = "[Paddle2ONNX]") {
    verbose_ = verbose;
    line_ = "";
    prefix_ = prefix;
  }

  template <typename T>
  P2OLogger& operator<<(const T& val) {
    if (!verbose_) {
      return *this;
    }
    std::stringstream ss;
    ss << val;
    line_ += ss.str();
    return *this;
  }
  P2OLogger& operator<<(std::ostream& (*os)(std::ostream&)) {
    if (!verbose_) {
      return *this;
    }
    std::cout << prefix_ << " " << line_ << std::endl;
    line_ = "";
    return *this;
  }
  ~P2OLogger() {
    if (!verbose_ && line_ != "") {
      std::cout << line_ << std::endl;
    }
  }

 private:
  std::string line_;
  std::string prefix_;
  bool verbose_ = true;
};

}  // namespace paddle2onnx
