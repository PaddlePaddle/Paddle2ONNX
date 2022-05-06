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

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace deploykit {

class KitLogger {
 public:
  KitLogger() {
    line_ = "";
    prefix_ = "[DeployKit]";
    verbose_ = true;
  }
  explicit KitLogger(bool verbose, const std::string& prefix = "[DeployKit]");

  template <typename T>
  KitLogger& operator<<(const T& val);
  KitLogger& operator<<(std::ostream& (*os)(std::ostream&));
  ~KitLogger() {
    if (!verbose_ && line_ != "") {
      std::cout << line_ << std::endl;
    }
  }

 private:
  std::string line_;
  std::string prefix_;
  bool verbose_ = true;
};

template <typename T>
KitLogger& KitLogger::operator<<(const T& val) {
  if (!verbose_) {
    return *this;
  }
  std::stringstream ss;
  ss << val;
  line_ += ss.str();
  return *this;
}

void Assert(bool condition, const std::string& message);
}  // namespace deploykit
