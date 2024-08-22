#include "paddle2onnx/parser/tensor_utils.h"
#include "paddle/pir/include/core/value.h"
#include "paddle/pir/include/core/program.h"
#include <map>
namespace paddle2onnx {
class PaddlePirParser {
public:
    bool Init(const std::string& _model, const std::string& _params = "");
    std::map<pir::Value, Weight> params;
    std::shared_ptr<pir::Program> pir_program_;
private:
    bool LoadProgram(const std::string& model);
    bool LoadParams(const std::string& path);
    bool GetParamValues(std::vector<pir::Value>* pir_values);

};
}  // namespace paddle2onnx
