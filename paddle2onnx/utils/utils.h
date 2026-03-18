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

namespace paddle2onnx {

inline void Assert(bool condition, const std::string& message) {
  if (!condition) {
    fprintf(stderr, "[ERROR][Paddle2ONNX] %s\n", message.c_str());
    std::abort();
  }
}

inline const std::string RequireOpset(const int32_t& opset_version) {
  return "Requires the minimal opset version of " +
         std::to_string(opset_version) + ".";
}

class P2OLogger {
 public:
  explicit P2OLogger(bool verbose = true, std::string prefix = "[Paddle2ONNX]")
      : verbose_(verbose), prefix_(std::move(prefix)) {}

  // Stream anything
  template <typename T>
  P2OLogger& operator<<(const T& value) {
    if (verbose_) {
      stream_ << value;
    }
    return *this;
  }

  // Support std::endl / manipulators (optional)
  P2OLogger& operator<<(std::ostream& (*manip)(std::ostream&)) {
    if (verbose_) {
      manip(stream_);
    }
    return *this;
  }

  // RAII: print on scope exit
  ~P2OLogger() {
    if (!verbose_) return;

    const std::string msg = stream_.str();
    if (!msg.empty()) {
      std::cout << prefix_ << " " << msg << std::endl;
    }
  }

  // Non-copyable (avoid double-print)
  P2OLogger(const P2OLogger&) = delete;
  P2OLogger& operator=(const P2OLogger&) = delete;

  // Movable (allows temporaries)
  P2OLogger(P2OLogger&&) = default;
  P2OLogger& operator=(P2OLogger&&) = default;

 private:
  bool verbose_;
  std::string prefix_;
  std::ostringstream stream_;
};
}  // namespace paddle2onnx
