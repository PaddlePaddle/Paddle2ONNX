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
#include <fstream>
#include <map>
#include <string>

#include "paddle2onnx/parser/pir_parser.h"
#include "paddle2onnx/utils/utils.h"
// This code is modified from
// https://blog.csdn.net/ZJU_fish1996/article/details/86515711
namespace paddle2onnx {
class Mapper;
class PaddleParser;
class OnnxHelper;
#define REGISTER_MAPPER(op_name, class_name)                            \
  class op_name##Generator : public Generator {                         \
   public:                                                              \
    op_name##Generator() { MapperHelper::Get()->Push(#op_name, this); } \
    void Touch() {}                                                     \
    Mapper* Create(const PaddleParser& p,                               \
                   OnnxHelper* h,                                       \
                   int64_t b,                                           \
                   int64_t o) {                                         \
      auto m = new class_name(p, h, b, o);                              \
      m->name_ = #class_name;                                           \
      return m;                                                         \
    }                                                                   \
  };                                                                    \
  op_name##Generator* op_name##inst = new op_name##Generator();         \
  int Touch##op_name##class_name() {                                    \
    op_name##inst->Touch();                                             \
    return 0;                                                           \
  }

#define REGISTER_PIR_MAPPER(op_name, class_name)                           \
  class op_name##PirGenerator : public PirGenerator {                      \
   public:                                                                 \
    op_name##PirGenerator() { MapperHelper::Get()->Push(#op_name, this); } \
    void Touch() {}                                                        \
    Mapper* Create(const PaddlePirParser& p,                               \
                   OnnxHelper* h,                                          \
                   int64_t i,                                              \
                   bool c) {                                               \
      auto m = new class_name(p, h, i, c);                                 \
      m->name_ = #class_name;                                              \
      return m;                                                            \
    }                                                                      \
  };                                                                       \
  op_name##PirGenerator* op_name##Pirinst = new op_name##PirGenerator();   \
  int TouchPir##op_name##class_name() {                                    \
    op_name##Pirinst->Touch();                                             \
    return 0;                                                              \
  }

class Generator {
 public:
  virtual Mapper* Create(const PaddleParser&,
                         OnnxHelper* helper,
                         int64_t,
                         int64_t) = 0;
};

class PirGenerator {
 public:
  virtual Mapper* Create(const PaddlePirParser&,
                         OnnxHelper* helper,
                         int64_t op_idx,
                         bool if_in_subblock) = 0;
};

class MapperHelper {
 private:
  std::map<std::string, Generator*> mappers;
  std::map<std::string, PirGenerator*> pir_mappers;
  std::map<std::string, int64_t> name_counter;
  MapperHelper() {}

 public:
  static MapperHelper* helper;
  static MapperHelper* Get() {
    if (nullptr == helper) {
      helper = new MapperHelper();
    }
    return helper;
  }

  std::vector<std::string> GetAllOps(bool isPir) {
    std::vector<std::string> operators;
    if (isPir) {
      for (auto iter = pir_mappers.begin(); iter != pir_mappers.end(); iter++) {
        operators.push_back(iter->first);
      }
    } else {
      for (auto iter = mappers.begin(); iter != mappers.end(); iter++) {
        operators.push_back(iter->first);
      }
    }
    return operators;
  }

  bool IsRegistered(const std::string& op_name) {
    return mappers.find(op_name) != mappers.end();
  }

  bool IsRegisteredInPir(const std::string& op_name, bool verbose) {
    static std::set<std::string> log_infos;
    auto iter_pir = pir_mappers.find(op_name);
    if (pir_mappers.end() != iter_pir) {
      return true;
    }
    auto iter = mappers.find(op_name);
    if (mappers.end() != iter) {
      std::string log_info =
          op_name +
          " is not registered in new ir mappers, but found in old ir mappers.";
      if (!log_infos.count(log_info)) {
        log_infos.insert(log_info);
        P2OLogger(verbose) << log_info << std::endl;
      }
      return false;
    }
    return false;
  }

  std::string GenName(const std::string& op_name) {
    std::string key = op_name + ".";
    if (name_counter.find(key) == name_counter.end()) {
      name_counter[key] = 0;
    } else {
      name_counter[key] += 1;
    }
    return key + std::to_string(name_counter[key]);
  }

  void ClearNameCounter() { name_counter.clear(); }

  Mapper* CreateMapper(const std::string& name,
                       const PaddleParser& parser,
                       OnnxHelper* helper,
                       int64_t block_id,
                       int64_t op_id) {
    Assert(mappers.find(name) != mappers.end(),
           name + " cannot be found in registered mappers.");
    return mappers[name]->Create(parser, helper, block_id, op_id);
  }

  Mapper* CreateMapper(const std::string& name,
                       const PaddlePirParser& pir_parser,
                       OnnxHelper* helper,
                       int64_t i,
                       bool if_in_subblock) {
    Assert(pir_mappers.find(name) != pir_mappers.end(),
           name + " can not be found in registered mappers.");
    return pir_mappers[name]->Create(pir_parser, helper, i, if_in_subblock);
  }

  void Push(const std::string& name, Generator* generator) {
    Assert(mappers.find(name) == mappers.end(),
           name + " has been registered before.");
    mappers[name] = generator;
  }

  void Push(const std::string& name, PirGenerator* pir_generator) {
    Assert(pir_mappers.find(name) == pir_mappers.end(),
           name + " has been registered before.");
    pir_mappers[name] = pir_generator;
  }
};
}  // namespace paddle2onnx
