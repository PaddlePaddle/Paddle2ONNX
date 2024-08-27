// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle2onnx/parser/pir_parser.h"
#include "paddle/fluid/pir/serialize_deserialize/include/interface.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/pir/include/core/builtin_dialect.h"
#include "paddle/phi/common/data_type.h"
#include <unordered_map>
#include "paddle2onnx/proto/p2o_paddle.pb.h"
#include "paddle/common/ddim.h"

std::unordered_map<phi::DataType, paddle2onnx::framework::proto::VarType_Type> pir_dtype_to_onnx_dtype = {
    {phi::DataType::FLOAT32, paddle2onnx::framework::proto::VarType_Type_FP32},
    {phi::DataType::INT64, paddle2onnx::framework::proto::VarType_Type_INT64},
    {phi::DataType::BOOL, paddle2onnx::framework::proto::VarType_Type_BOOL},
    {phi::DataType::FLOAT16, paddle2onnx::framework::proto::VarType_Type_FP16},
    {phi::DataType::BFLOAT16, paddle2onnx::framework::proto::VarType_Type_BF16},
    {phi::DataType::FLOAT64, paddle2onnx::framework::proto::VarType_Type_FP64},
    {phi::DataType::UINT8, paddle2onnx::framework::proto::VarType_Type_UINT8},
    {phi::DataType::INT32, paddle2onnx::framework::proto::VarType_Type_INT32},
    {phi::DataType::COMPLEX64, paddle2onnx::framework::proto::VarType_Type_COMPLEX64},
    {phi::DataType::COMPLEX128, paddle2onnx::framework::proto::VarType_Type_COMPLEX128},
    {phi::DataType::INT8, paddle2onnx::framework::proto::VarType_Type_INT8},
    {phi::DataType::INT16, paddle2onnx::framework::proto::VarType_Type_INT16},
};

phi::DataType TransToPhiDataType(pir::Type dtype) {
  if (dtype.isa<pir::BFloat16Type>()) {
    return phi::DataType::BFLOAT16;
  } else if (dtype.isa<pir::Float16Type>()) {
    return phi::DataType::FLOAT16;
  } else if (dtype.isa<pir::Float32Type>()) {
    return phi::DataType::FLOAT32;
  } else if (dtype.isa<pir::Float64Type>()) {
    return phi::DataType::FLOAT64;
  } else if (dtype.isa<pir::UInt8Type>()) {
    return phi::DataType::UINT8;
  } else if (dtype.isa<pir::Int8Type>()) {
    return phi::DataType::INT8;
  } else if (dtype.isa<pir::Int16Type>()) {
    return phi::DataType::INT16;
  } else if (dtype.isa<pir::Int32Type>()) {
    return phi::DataType::INT32;
  } else if (dtype.isa<pir::Int64Type>()) {
    return phi::DataType::INT64;
  } else if (dtype.isa<pir::IndexType>()) {
    return phi::DataType::INT32;
  } else if (dtype.isa<pir::BoolType>()) {
    return phi::DataType::BOOL;
  } else if (dtype.isa<pir::Complex64Type>()) {
    return phi::DataType::COMPLEX64;
  } else if (dtype.isa<pir::Complex128Type>()) {
    return phi::DataType::COMPLEX128;
  } else if (dtype.isa<pir::Float8E4M3FNType>()) {
    return phi::DataType::FLOAT8_E4M3FN;
  } else if (dtype.isa<pir::Float8E5M2Type>()) {
    return phi::DataType::FLOAT8_E5M2;
  } else {
      std::cerr << "Unsupported data type: " << dtype << std::endl;
  }
}

namespace paddle2onnx{
    bool PaddlePirParser::LoadProgram(const std::string& model) {
        pir::IrContext *ctx = pir::IrContext::Instance();
        ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
        ctx->GetOrRegisterDialect<pir::BuiltinDialect>();
        //pir::Program new_program(ctx);
        pir_program_ = std::make_shared<pir::Program>(ctx);
        if (!pir::ReadModule(model, pir_program_.get(),/*pir_version*/ 0)) {
            P2OLogger() << "Failed to deserialize PaddlePaddle model." << std::endl;
            return false;
        }
        return true;
    }
    bool PaddlePirParser::GetParamValueName(std::vector<std::string>* var_names){
        var_names->clear();
        P2OLogger()<<"Start getting paramas value name from pir::program" <<std::endl;
        auto global_block = pir_program_->block();
        std::vector<pir::Value> value_list;
        for(auto &op : global_block->ops()){
            if (op->name() == "builtin.parameter" && op->HasAttribute(kAttrIsPersistable)) {
                 auto attrs = op->attribute(kAttrIsPersistable)
                                  .dyn_cast<pir::ArrayAttribute>()
                                  .AsVector();
                for (uint32_t i = 0; i < attrs.size(); i++) {
                   bool is_persistable =
                       attrs[i].dyn_cast<pir::BoolAttribute>().data();
                   if (is_persistable) {
                        auto value = static_cast<pir::Value>(op->result(i));
                        if (auto param_op = value.defining_op<::pir::ParameterOp>()){
                            var_names->push_back(param_op.param_name());
                        }
                   }
                }
            }

        }

        std::sort(var_names->begin(),var_names->end());
        return true;
    }


    bool PaddlePirParser::LoadParams(const std::string& path) {
        params.clear();
        std::ifstream is(path, std::ios::in | std::ios::binary);
        if (!is.is_open()) {
            P2OLogger() << "Cannot open file " << path << " to read." << std::endl;
            return false;
        }
        is.seekg(0, std::ios::end);
        int64_t total_size = is.tellg();
        is.seekg(0, std::ios::beg);
        std::vector<std::string> var_names;
        GetParamValueName(&var_names);
        P2OLogger()<<"getting paramas value name from pir::program successfully" << std::endl;

        int64_t read_size = 0;
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
                    "Paddle2ONNX: Only support weight with lod_level = 0.");
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
                P2OLogger() << "Unexcepted situation happend while reading parameters "
                            "of PaddlePaddle pir model."
                            << std::endl;
                return false;
            }
            params[var_names[index]] = weight;
            }
        }
        is.close();
        return true;
    }

    bool PaddlePirParser::Init(const std::string& _model, const std::string& _params){
        std::vector<Weight> weights;
        if (!LoadProgram(_model)) {
            P2OLogger() << "Failed to load program of PaddlePaddle pir model ." << std::endl;
            return false;
        }
        P2OLogger() << "load PaddlePaddle pir model successfully ." << std::endl;
        if (_params != "") {
            if (!LoadParams(_params)) {
            P2OLogger() << "Failed to load parameters of PaddlePaddle model."
                        << std::endl;
            return false;
            }
        }

        // InitBlock();
        GetGlobalBlocksOps();
        GetGlobalBlockInputOutputInfo();
        return true;
    }
    int PaddlePirParser::NumOfBlocks() const {
        size_t num_blocks = 0;
        auto top_level_op = pir_program_->module_op();
        for (size_t i = 0; i < top_level_op->num_regions(); ++i) {
            auto &region = top_level_op->region(i);
            num_blocks += region.size();
        }
        return num_blocks;
    }

    int PaddlePirParser::NumOfProgramOps() const{
        return pir_program_->num_ops();
    }

    void PaddlePirParser::GetGlobalBlocksOps() {
        is_quantized_model=false;
        global_blocks_ops.clear();
        // _constant_ops.clear();
        // // global_blocks_ops.resize(NumOfBlocks());
        // _constant_ops.resize(NumOfBlocks());
        auto global_block = pir_program_->block();
        for (auto &op : global_block->ops()) {
            std::cout<<"op name: "<<op->name()<<std::endl;
            global_blocks_ops.push_back(op);
        }
    }
    TensorInfo PaddlePirParser::GetTensorInfo(std::string name, const pir::Operation* op){
        std::cout<<"op num results:"<<op->num_results()<<std::endl;
        if(op->result(0).type().isa<pir::DenseTensorType>()){
            TensorInfo info;
            //get info.name
            info.name = name;
            std::cout<<"info.name:"<<info.name<<std::endl;
            //get info.dtype
            auto type= op->result(0).type().cast<pir::DenseTensorType>().dtype();
            auto data_type = TransToPhiDataType(type);
            auto it = pir_dtype_to_onnx_dtype.find(data_type);
            if (it != pir_dtype_to_onnx_dtype.end()) {
              info.dtype = it->second;
              std::cout<<"info.dtype:"<<info.dtype<<std::endl;
            } else {
                std::cerr << "data_type not found" << std::endl;
            }
            //get info.shape
            std::vector<int64_t> dims = common::vectorize(op->result(0).type().cast<pir::DenseTensorType>().dims());
            for(auto dim:dims){
              std::cout<<"dim:"<<dim<<std::endl;
            }
            std::cout <<"----------"<<std::endl;
            info.shape = dims;
            // for (size_t i = 0; i < shape.size(); ++i) {
            //     info.shape.push_back(shape[i]);
            //     std::cout<<"info.shape:"<<info.shape[i]<<std::endl;
            // }
            return info;

        }else{
            std::cerr <<"only support dense tensor type"<<std::endl;
        }




    }


    void PaddlePirParser::GetGlobalBlockInputOutputInfo() {
        inputs.clear();
        outputs.clear();
        std::ostringstream print_stream;
        print_stream << "ForwardProgram is :\n";
        pir_program_->Print(print_stream);
        std::cout << "Program (fwd | bwd): \n" << print_stream.str() << std::endl;
        std::cout << "global_blocks_ops.size(): " << global_blocks_ops.size() << std::endl;
        for(auto op : global_blocks_ops){
            if(op->name()=="pd_op.data"){
                std::cout<<"op->name:"<<op->name()<<std::endl;
                std::string var_name = op->attribute<pir::StrAttribute>("name").AsString();
                std::cout<<"var_name:"<<var_name<<std::endl;
                inputs.push_back(GetTensorInfo(var_name,op));
            }else if(op->name()=="pd_op.fetch"){
                std::cout<<"op->name:"<<op->name()<<std::endl;
                std::string var_name = op->attribute<pir::StrAttribute>("name").AsString();
                std::cout<<"var_name:"<<var_name<<std::endl;
                outputs.push_back(GetTensorInfo(var_name,op));
            }
        }
    }
}
