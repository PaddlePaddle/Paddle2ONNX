#include "paddle2onnx/parser/pir_parser.h"
#include "paddle/fluid/pir/serialize_deserialize/include/interface.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/pir/include/core/builtin_dialect.h"

namespace paddle2onnx
{
    bool PaddlePirParser::LoadProgram(const std::string& model) {
        pir_program_ = std::make_shared<pir::Program>(pir::IrContext::Instance());
        ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
        ctx->GetOrRegisterDialect<pir::BuiltinDialect>();
        pir::Program new_program(ctx);
        if (!pir::ReadModule(model, &pir_program_,/*pir_version*/ 0)) {
            P2OLogger() << "Failed to deserialize PaddlePaddle model." << std::endl;
            return false;
        }
        return true;
    }
    bool GetParamValues(std::vector<pir::Value>* pir_values){
        pir_values->clear();
        //如何从pir::program中获取所有的value
        
    }
    bool PaddleParser::LoadParams(const std::string& path) {
        params.clear();
        std::ifstream is(path, std::ios::in | std::ios::binary);
        if (!is.is_open()) {
            P2OLogger() << "Cannot open file " << path << " to read." << std::endl;
            return false;
        }
        is.seekg(0, std::ios::end);
        int64_t total_size = is.tellg();
        is.seekg(0, std::ios::beg);
        std::vector<pir::Value> pir_values;
        GetParamValues(&pir_values);

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
                            "of PaddlePaddle model."
                            << std::endl;
                return false;
            }
            params[var_names[index]] = weight;
            }
        }
        for(const auto& pair:params){
                std::cout << "Key:"<<pair.first << ",Dtype: "<<pair.second.dtype << std::endl;
                for(auto i:pair.second.shape){
                std::cout << i<<",";
                }
                std::cout << std::endl;
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
    // if (_params != "") {
        // if (!LoadParams(_params)) {
        // P2OLogger() << "Failed to load parameters of PaddlePaddle model."
        //             << std::endl;
        // return false;
        // }
    // }
    // InitBlock();
    return true;
    }
} // namespace paddle2onnx
