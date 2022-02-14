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

#include "paddle2onnx/parser/parser.h"
#include <fstream>
#include <string>
#include "paddle2onnx/utils/utils.h"

namespace paddle2onnx {
bool PaddleParser::LoadProgram(const std::string& path) {
  std::ifstream fin(path, std::ios::in | std::ios::binary);
  if (!fin.is_open()) {
    std::cerr << "Fail to read model file " << path
              << ", please make sure your model file or file path is valid."
              << std::endl;
    return false;
  }

  std::string contents;
  fin.seekg(0, std::ios::end);
  contents.clear();
  contents.resize(fin.tellg());
  fin.seekg(0, std::ios::beg);
  fin.read(&(contents.at(0)), contents.size());
  fin.close();

  prog = std::make_shared<paddle2onnx::framework::proto::ProgramDesc>();
  prog->ParseFromString(contents);
  return true;
}

bool PaddleParser::GetParamNames(std::vector<std::string>* var_names) {
  var_names->clear();
  int block_size = prog->blocks_size();
  for (auto i = 0; i < block_size; ++i) {
    auto block = prog->blocks(i);
    int vars_size = block.vars_size();
    for (auto j = 0; j < vars_size; ++j) {
      auto type = block.vars(j).type().type();
      if (type == framework::proto::VarType_Type::VarType_Type_SELECTED_ROWS) {
        std::cerr << "VarType of SELECTED_ROWS is not supported by Paddle2ONNX."
                  << std::endl;
        return false;
      }
      if (type == framework::proto::VarType_Type::VarType_Type_FEED_MINIBATCH) {
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
      var_names->push_back(block.vars(j).name());
    }
  }
  std::sort(var_names->begin(), var_names->end());
  return true;
}

bool PaddleParser::LoadParams(const std::string& path) {
  params.clear();
  std::ifstream is(path, std::ios::in | std::ios::binary);
  if (!is.is_open()) {
    std::cerr << "Cannot open file " << path << " to read." << std::endl;
    return false;
  }
  is.seekg(0, std::ios::end);
  int total_size = is.tellg();
  is.seekg(0, std::ios::beg);

  std::vector<std::string> var_names;
  GetParamNames(&var_names);

  int read_size = 0;
  while (read_size < total_size) {
    {
      // read version, we don't need this
      uint32_t version;
      read_size += sizeof(version);
      is.read(reinterpret_cast<char*>(&version), sizeof(version));
    }
    {
      // read lod_level, we don't use it
      // this has to be zero, otherwise not support
      uint64_t lod_level;
      read_size += sizeof(lod_level);
      is.read(reinterpret_cast<char*>(&lod_level), sizeof(lod_level));
      Assert(lod_level == 0,
             "Paddle2ONNX: Only support weight with lod_level = 1.");
    }
    {
      // Another version, we don't use it
      uint32_t version;
      read_size += sizeof(version);
      is.read(reinterpret_cast<char*>(&version), sizeof(version));
    }
    {
      // read size of TensorDesc
      int32_t size;
      read_size += sizeof(size);
      is.read(reinterpret_cast<char*>(&size), sizeof(size));
      // read TensorDesc
      std::unique_ptr<char[]> buf(new char[size]);
      read_size += size;
      is.read(reinterpret_cast<char*>(buf.get()), size);

      std::unique_ptr<paddle2onnx::framework::proto::VarType_TensorDesc>
          tensor_desc(new paddle2onnx::framework::proto::VarType_TensorDesc());
      tensor_desc->ParseFromArray(buf.get(), size);

      Weight weight;

      int32_t numel = 1;
      int32_t data_type = tensor_desc->data_type();
      weight.dtype = data_type;
      for (auto i = 0; i < tensor_desc->dims().size(); ++i) {
        numel *= tensor_desc->dims()[i];
        weight.shape.push_back(tensor_desc->dims()[i]);
      }

      // read weight data
      weight.buffer.resize(numel * PaddleDataTypeSize(data_type));
      read_size += numel * PaddleDataTypeSize(data_type);
      is.read(weight.buffer.data(), numel * PaddleDataTypeSize(data_type));
      auto index = params.size();
      if (index >= var_names.size()) {
        std::cerr << "Paddle2ONNX: Unexcept situation happened, index should "
                     "less than size of var_names."
                  << std::endl;
        return false;
      }
      params[var_names[index]] = weight;
    }
  }
  is.close();
  return true;
}

int PaddleParser::NumOfBlocks() const { return prog->blocks_size(); }

int PaddleParser::NumOfOps(int block_idx) const {
  Assert(block_idx < NumOfBlocks(),
         "block_idx is greater than number of blocks.");
  return prog->blocks(block_idx).ops_size();
}

const framework::proto::OpDesc PaddleParser::GetOpDesc(int32_t block_idx,
                                                       int32_t op_idx) const {
  Assert(block_idx < NumOfBlocks(),
         "block_idx is greater than number of blocks.");
  Assert(op_idx < NumOfOps(block_idx),
         "op_idx is greater than number of operators.");
  return prog->blocks(block_idx).ops(op_idx);
}

// Sometimes the model contains no parameters
// In this case, we only need the model_file
bool PaddleParser::Init(const std::string& _model_filename) {
  return Init(_model_filename, "");
}

bool PaddleParser::Init(const std::string& _model_filename,
                        const std::string& _params_filename) {
  std::vector<Weight> weights;
  if (!LoadProgram(_model_filename)) {
    std::cerr << "Paddle2ONNX: Load program failed!." << std::endl;
    return false;
  }
  if (_params_filename != "") {
    if (!LoadParams(_params_filename)) {
      std::cerr << "Paddle2ONNX: Load parameters failed!." << std::endl;
      return false;
    }
  } else {
    std::cerr << "[WARNING] You haven't set a params file, this only valid "
                 "while the model has no weights."
              << std::endl;
  }

  GetBlocksVarName2Id();
  GetBlocksOps();
  GetGlobalBlockInputOutputInfo();
  return true;
}

void PaddleParser::GetBlocksVarName2Id() {
  _blocks_var_name2id.clear();
  _blocks_var_name2id.resize(prog->blocks_size());
  for (auto i = 0; i < prog->blocks_size(); ++i) {
    for (auto j = 0; j < prog->blocks(i).vars_size(); ++j) {
      _blocks_var_name2id[i][prog->blocks(i).vars(j).name()] = j;
    }
  }
}

void PaddleParser::GetBlocksOps() {
  _blocks_ops.clear();
  _blocks_ops.resize(prog->blocks_size());
  for (auto i = 0; i < prog->blocks_size(); ++i) {
    _blocks_ops[i].reserve(prog->blocks(i).ops_size());
    for (auto j = 0; j < prog->blocks(i).ops_size(); ++j) {
      _blocks_ops[i].push_back(&prog->blocks(i).ops(j));
    }
  }
}

TensorInfo PaddleParser::GetTensorInfo(
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

bool PaddleParser::OpHasInput(int64_t block_id, int64_t op_id,
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

std::vector<TensorInfo> PaddleParser::GetOpInput(
    int64_t block_id, int64_t op_id, const std::string& name) const {
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
  Assert(found, "Cannot find input:" + name + " in operator: " + op.type());
  return inputs;
}

bool PaddleParser::OpHasOutput(int64_t block_id, int64_t op_id,
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

std::vector<TensorInfo> PaddleParser::GetOpOutput(
    int64_t block_id, int64_t op_id, const std::string& name) const {
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
  Assert(found, "Cannot find output:" + name + " in operator: " + op.type());
  return outputs;
}

std::string PaddleParser::GetOpAttrType(
    const paddle2onnx::framework::proto::OpDesc& op,
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

bool PaddleParser::OpHasAttr(const paddle2onnx::framework::proto::OpDesc& op,
                             const std::string& name) const {
  bool found = false;
  for (auto i = 0; i < op.attrs_size(); ++i) {
    // set found to true when name is in op attrs and can use GetOpAttr to get
    // value
    if (op.attrs(i).name() == name && GetOpAttrType(op, name) != "NOTFOUND") {
      found = true;
      break;
    }
  }
  return found;
}

void PaddleParser::GetOpAttr(const paddle2onnx::framework::proto::OpDesc& op,
                             const std::string name, int64_t* res) const {
  bool found = false;
  for (auto i = 0; i < op.attrs_size(); ++i) {
    if (op.attrs(i).name() == name) {
      found = true;
      Assert(op.attrs(i).has_i() || op.attrs(i).has_l(),
             "Cannot find int32/int64 data from attr:" + name + " in op:" +
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

void PaddleParser::GetOpAttr(const paddle2onnx::framework::proto::OpDesc& op,
                             const std::string name, float* res) const {
  bool found = false;
  for (auto i = 0; i < op.attrs_size(); ++i) {
    if (op.attrs(i).name() == name) {
      found = true;
      Assert(op.attrs(i).has_f(), "Cannot find float data from attr:" + name +
                                      " in op:" + op.type());
      *res = op.attrs(i).f();
      break;
    }
  }
  Assert(found, "Cannot found attribute " + name + " in op: " + op.type());
}

void PaddleParser::GetOpAttr(const paddle2onnx::framework::proto::OpDesc& op,
                             const std::string name, bool* res) const {
  bool found = false;
  for (auto i = 0; i < op.attrs_size(); ++i) {
    if (op.attrs(i).name() == name) {
      found = true;
      Assert(op.attrs(i).has_b(),
             "Cannot find bool data from attr:" + name + " in op:" + op.type());
      *res = op.attrs(i).b();
      break;
    }
  }
  Assert(found, "Cannot found attribute " + name + " in op: " + op.type());
}

void PaddleParser::GetOpAttr(const paddle2onnx::framework::proto::OpDesc& op,
                             const std::string name, std::string* res) const {
  bool found = false;
  for (auto i = 0; i < op.attrs_size(); ++i) {
    if (op.attrs(i).name() == name) {
      found = true;
      Assert(op.attrs(i).has_s(), "Cannot find string data from attr:" + name +
                                      " in op:" + op.type());
      *res = op.attrs(i).s();
      break;
    }
  }
  Assert(found, "Cannot found attribute " + name + " in op: " + op.type());
}

void PaddleParser::GetOpAttr(const paddle2onnx::framework::proto::OpDesc& op,
                             const std::string name,
                             std::vector<int64_t>* res) const {
  bool found = false;
  res->clear();
  for (auto i = 0; i < op.attrs_size(); ++i) {
    if (op.attrs(i).name() == name) {
      Assert(op.attrs(i).ints_size() > 0 || op.attrs(i).longs_size() > 0,
             "Cannot find list of int32/int64 data from attr:" + name +
                 "in op:" + op.type());
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

void PaddleParser::GetOpAttr(const paddle2onnx::framework::proto::OpDesc& op,
                             const std::string name,
                             std::vector<float>* res) const {
  bool found = false;
  res->clear();
  for (auto i = 0; i < op.attrs_size(); ++i) {
    if (op.attrs(i).name() == name) {
      Assert(op.attrs(i).floats_size() > 0,
             "Cannot find list of float data from attr:" + name + "in op:" +
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

void PaddleParser::GetGlobalBlockInputOutputInfo() {
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

int32_t PaddleDataTypeSize(int32_t paddle_dtype) {
  Assert(paddle_dtype != FP16, "Float16 is not supported.");
  if (paddle_dtype == P2ODataType::BOOL) {
    return sizeof(bool);
  } else if (paddle_dtype == P2ODataType::INT16) {
    return sizeof(int16_t);
  } else if (paddle_dtype == P2ODataType::INT32) {
    return sizeof(int32_t);
  } else if (paddle_dtype == P2ODataType::INT64) {
    return sizeof(int64_t);
  } else if (paddle_dtype == P2ODataType::FP32) {
    return sizeof(float);
  } else if (paddle_dtype == P2ODataType::FP64) {
    return sizeof(double);
  } else {
    Assert(false, "Unexpected data type");
  }
  return -1;
}

}  // namespace paddle2onnx
