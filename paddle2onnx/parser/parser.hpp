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
#include <algorithm>
#include <cassert>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>
#include "paddle2onnx/parser/parse_params.hpp"
#include "paddle2onnx/parser/parse_program.hpp"

namespace paddle2onnx {

struct TensorInfo {
  std::string name;
  std::vector<int64_t> shape;
  int64_t Rank() { return static_cast<int64_t>(shape.size()); }
  int32_t dtype;
};

struct PaddleParser {
  // recording variable name:id for each block of a program
  std::vector<std::map<std::string, int32_t>> _blocks_var_name2id;
  // recoring set of operators for each block
  std::vector<std::vector<const paddle2onnx::framework::proto::OpDesc*>>
      _blocks_ops;
  std::shared_ptr<paddle2onnx::framework::proto::ProgramDesc> prog;
  std::map<std::string, Weight> params;
  std::vector<TensorInfo> inputs;
  std::vector<TensorInfo> outputs;

  int NumOfBlocks() const { return prog->blocks_size(); }

  int NumOfOps(int block_idx) const {
    Assert(block_idx < NumOfBlocks(),
           "block_idx is greater than number of blocks.");
    return prog->blocks(block_idx).ops_size();
  }

  const framework::proto::OpDesc GetOpDesc(int32_t block_idx,
                                           int32_t op_idx) const {
    Assert(block_idx < NumOfBlocks(),
           "block_idx is greater than number of blocks.");
    Assert(op_idx < NumOfOps(block_idx),
           "op_idx is greater than number of operators.");
    return prog->blocks(block_idx).ops(op_idx);
  }

  // Sometimes the model contains no parameters
  // In this case, we only need the model_file
  void Init(const std::string& _model_filename) { Init(_model_filename, ""); }

  void Init(const std::string& _model_filename,
            const std::string& _params_filename) {
    std::vector<Weight> weights;
    prog = LoadProgram(_model_filename);
    if (_params_filename != "") {
      LoadParams(_params_filename, &weights);
    } else {
      std::cerr << "[WARNING] You haven't set a params file, this only valid "
                   "while the model has no weights."
                << std::endl;
    }

    GetBlocksVarName2Id();
    GetBlocksOps();
    GetGlobalBlockInputOutputInfo();

    std::vector<std::string> var_names;
    int block_size = prog->blocks_size();
    for (auto i = 0; i < block_size; ++i) {
      auto block = prog->blocks(i);
      int vars_size = block.vars_size();
      for (auto j = 0; j < vars_size; ++j) {
        auto type = block.vars(j).type().type();
        Assert(
            type != framework::proto::VarType_Type::VarType_Type_SELECTED_ROWS,
            "VarType of SELECTED_ROWS is not supported by Paddle2ONNX.");
        if (type ==
            framework::proto::VarType_Type::VarType_Type_FEED_MINIBATCH) {
          continue;
        }
        if (type == paddle2onnx::framework::proto::VarType_Type::
                        VarType_Type_FETCH_LIST) {
          continue;
        }
        if (type ==
            paddle2onnx::framework::proto::VarType_Type::VarType_Type_READER) {
          continue;
        }
        if (type ==
            paddle2onnx::framework::proto::VarType_Type::VarType_Type_RAW) {
          continue;
        }
        if (!block.vars(j).persistable()) {
          continue;
        }
        var_names.push_back(block.vars(j).name());
      }
    }
    std::sort(var_names.begin(), var_names.end());
    params.clear();
    Assert(var_names.size() == weights.size(),
           "Number of read parameters is not same with number of variables in "
           "program.");
    for (size_t i = 0; i < var_names.size(); ++i) {
      params[var_names[i]] = weights[i];
    }
  }

  void GetBlocksVarName2Id() {
    _blocks_var_name2id.clear();
    _blocks_var_name2id.resize(prog->blocks_size());
    for (auto i = 0; i < prog->blocks_size(); ++i) {
      for (auto j = 0; j < prog->blocks(i).vars_size(); ++j) {
        _blocks_var_name2id[i][prog->blocks(i).vars(j).name()] = j;
      }
    }
  }

  void GetBlocksOps() {
    _blocks_ops.clear();
    _blocks_ops.resize(prog->blocks_size());
    for (auto i = 0; i < prog->blocks_size(); ++i) {
      _blocks_ops[i].reserve(prog->blocks(i).ops_size());
      for (auto j = 0; j < prog->blocks(i).ops_size(); ++j) {
        _blocks_ops[i].push_back(&prog->blocks(i).ops(j));
      }
    }
  }

  TensorInfo GetTensorInfo(
      const std::string& name,
      const paddle2onnx::framework::proto::BlockDesc& block) const {
    auto block_idx = block.idx();
    auto iter = _blocks_var_name2id[block_idx].find(name);
    Assert(_blocks_var_name2id[block_idx].end() != iter,
           "Cannot find " + name + " in _blocks_var_name2id.");
    auto var_idx = iter->second;

    auto tensor = block.vars(var_idx).type().lod_tensor();
    TensorInfo info;
    info.name = name;
    info.dtype = tensor.tensor().data_type();
    for (auto i = 0; i < tensor.tensor().dims_size(); ++i) {
      info.shape.push_back(tensor.tensor().dims(i));
    }
    return info;
  }

  bool OpHasInput(int64_t block_id, int64_t op_id,
                  const std::string& name) const {
    auto block = prog->blocks(block_id);
    auto op = block.ops(op_id);
    for (auto i = 0; i < op.inputs_size(); ++i) {
      if (op.inputs(i).parameter() == name) {
        if (op.inputs(i).arguments_size() > 0) {
          return true;
        }
      }
    }
    return false;
  }

  std::vector<TensorInfo> GetOpInput(int64_t block_id, int64_t op_id,
                                     const std::string& name) const {
    auto block = prog->blocks(block_id);
    auto op = block.ops(op_id);
    std::vector<TensorInfo> inputs;
    bool found = false;
    for (auto i = 0; i < op.inputs_size(); ++i) {
      if (op.inputs(i).parameter() == name) {
        for (auto j = 0; j < op.inputs(i).arguments_size(); ++j) {
          inputs.push_back(GetTensorInfo(op.inputs(i).arguments(j), block));
          found = true;
        }
        break;
      }
    }
    Assert(found, "Cannot find input: " + name + " in operator: " + op.type());
    return inputs;
  }

  bool OpHasOutput(int64_t block_id, int64_t op_id,
                   const std::string& name) const {
    auto block = prog->blocks(block_id);
    auto op = block.ops(op_id);
    for (auto i = 0; i < op.outputs_size(); ++i) {
      if (op.outputs(i).parameter() == name) {
        if (op.outputs(i).arguments_size() > 0) {
          return true;
        }
      }
    }
    return false;
  }

  std::vector<TensorInfo> GetOpOutput(int64_t block_id, int64_t op_id,
                                      const std::string& name) const {
    auto block = prog->blocks(block_id);
    auto op = block.ops(op_id);
    std::vector<TensorInfo> outputs;
    bool found = false;
    for (auto i = 0; i < op.outputs_size(); ++i) {
      if (op.outputs(i).parameter() == name) {
        for (auto j = 0; j < op.outputs(i).arguments_size(); ++j) {
          outputs.push_back(GetTensorInfo(op.outputs(i).arguments(j), block));
          found = true;
        }
        break;
      }
    }
    Assert(found, "Cannot find output: " + name + " in operator: " + op.type());
    return outputs;
  }

  bool OpHasAttr(const paddle2onnx::framework::proto::OpDesc& op,
                 const std::string& name) const {
    bool found = false;
    for (auto i = 0; i < op.attrs_size(); ++i) {
      if (op.attrs(i).name() == name && GetOpAttrType(op, name) != "NOTFOUND") {
        found = true;
        break;
      }
    }
    return found;
  }

  std::string GetOpAttrType(const paddle2onnx::framework::proto::OpDesc& op,
                            const std::string& name) const {
    std::string type = "NOTFOUND";
    for (auto i = 0; i < op.attrs_size(); ++i) {
      if (op.attrs(i).name() == name) {
        if (op.attrs(i).has_i() || op.attrs(i).has_l()) type = "INT64";
        if (op.attrs(i).has_f()) type = "FLOAT";
        if (op.attrs(i).has_b()) type = "BOOL";
        if (op.attrs(i).has_s()) type = "STRING";
        if (op.attrs(i).ints_size() > 0 || op.attrs(i).longs_size() > 0)
          type = "INT64_LIST";
        if (op.attrs(i).floats_size() > 0) type = "FLOAT_LIST";
        break;
      }
    }
    return type;
  }

  void GetOpAttr(const paddle2onnx::framework::proto::OpDesc& op,
                 const std::string name, int64_t* res) const {
    bool found = false;
    for (auto i = 0; i < op.attrs_size(); ++i) {
      if (op.attrs(i).name() == name) {
        found = true;
        Assert(op.attrs(i).has_i() || op.attrs(i).has_l(),
               "Cannot find int32/int64 data from attr: " + name + " in op: " +
                   op.type());
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

  void GetOpAttr(const paddle2onnx::framework::proto::OpDesc& op,
                 const std::string name, float* res) const {
    bool found = false;
    for (auto i = 0; i < op.attrs_size(); ++i) {
      if (op.attrs(i).name() == name) {
        found = true;
        Assert(op.attrs(i).has_f(), "Cannot find float data from attr: " +
                                        name + " in op: " + op.type());
        *res = op.attrs(i).f();
        break;
      }
    }
    Assert(found, "Cannot found attribute " + name + " in op: " + op.type());
  }

  void GetOpAttr(const paddle2onnx::framework::proto::OpDesc& op,
                 const std::string name, bool* res) const {
    bool found = false;
    for (auto i = 0; i < op.attrs_size(); ++i) {
      if (op.attrs(i).name() == name) {
        found = true;
        Assert(op.attrs(i).has_b(), "Cannot find bool data from attr: " + name +
                                        " in op: " + op.type());
        *res = op.attrs(i).b();
        break;
      }
    }
    Assert(found, "Cannot found attribute " + name + " in op: " + op.type());
  }

  void GetOpAttr(const paddle2onnx::framework::proto::OpDesc& op,
                 const std::string name, std::string* res) const {
    bool found = false;
    for (auto i = 0; i < op.attrs_size(); ++i) {
      if (op.attrs(i).name() == name) {
        found = true;
        Assert(op.attrs(i).has_s(), "Cannot find string data from attr: " +
                                        name + " in op: " + op.type());
        *res = op.attrs(i).s();
        break;
      }
    }
    Assert(found, "Cannot found attribute " + name + " in op: " + op.type());
  }

  void GetOpAttr(const paddle2onnx::framework::proto::OpDesc& op,
                 const std::string name, std::vector<int64_t>* res) const {
    bool found = false;
    res->clear();
    for (auto i = 0; i < op.attrs_size(); ++i) {
      if (op.attrs(i).name() == name) {
        Assert(op.attrs(i).ints_size() > 0 || op.attrs(i).longs_size() > 0,
               "Cannot find list of int32/int64 data from attr: " + name +
                   " in op: " + op.type());
        found = true;
        if (op.attrs(i).ints_size() > 0) {
          for (auto j = 0; j < op.attrs(i).ints_size(); ++j) {
            res->push_back((int64_t)(op.attrs(i).ints(j)));
          }
        } else {
          for (auto j = 0; j < op.attrs(i).longs_size(); ++j) {
            res->push_back(op.attrs(i).longs(j));
          }
        }
        break;
      }
    }
    Assert(found, "Cannot found attribute " + name + " in op: " + op.type());
  }

  void GetOpAttr(const paddle2onnx::framework::proto::OpDesc& op,
                 const std::string name, std::vector<float>* res) const {
    bool found = false;
    res->clear();
    for (auto i = 0; i < op.attrs_size(); ++i) {
      if (op.attrs(i).name() == name) {
        Assert(op.attrs(i).floats_size() > 0,
               "Cannot find list of float data from attr: " + name + "in op: " +
                   op.type());
        found = true;
        for (auto j = 0; j < op.attrs(i).floats_size(); ++j) {
          res->push_back((int64_t)(op.attrs(i).floats(j)));
        }
        break;
      }
    }
    Assert(found, "Cannot found attribute " + name + " in op: " + op.type());
  }

  void GetGlobalBlockInputOutputInfo() {
    inputs.clear();
    outputs.clear();
    for (auto i = 0; i < prog->blocks(0).ops_size(); ++i) {
      if (prog->blocks(0).ops(i).type() == "fetch") {
        std::string name = prog->blocks(0).ops(i).inputs(0).arguments(0);
        outputs.push_back(GetTensorInfo(name, prog->blocks(0)));
      } else if (prog->blocks(0).ops(i).type() == "feed") {
        std::string name = prog->blocks(0).ops(i).outputs(0).arguments(0);
        inputs.push_back(GetTensorInfo(name, prog->blocks(0)));
      }
    }
  }
};

}  // namespace paddle2onnx
