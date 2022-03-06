# Paddle2ONNX开发指南
本文档为Paddle2ONNX的新OP开发指南，旨在帮助开发者快速掌握Paddle2ONNX的新OP开发方式，在遇到模型转换失败时能有应对方法，快速解决自己遇到的问题。
## Paddle2ONNX简介 ##
ONNX (Open Neural Network Exchange) 是针对机器学习所设计的开源文件格式，用于存储训练好的模型。它使得不同的人工智能框架可以采用相同格式存储模型并交互。通过ONNX格式，Paddle模型可以使用OpenVINO、ONNX Runtime和TensorRT等推理框架进行推理。
Paddle2ONNX是PaddlePaddle的工具套件，负责将Paddle的inference模型转换为ONNX格式，便于开发者将Paddle模型扩展到支持ONNX部署的框架上进行推理。
## Paddle2ONNX新OP开发步骤 ##
Paddle2ONNX开发的主要步骤为：  
1.根据Paddle OP查阅对应的Paddle API并掌握其使用方法；  
2.根据Paddle OP的原理通过ONNX OP直接或者组合实现相同功能；  
3.为Paddle OP的转换添加单测；  
4.为Paddle2ONNX提PR；  
### 查找Paddle API和paddle OP对应关系 ###
Paddle OP的转换需要掌握Paddle OP的原理和使用方式，因此需要查阅Paddle OP对应的Paddle API。  
当遇到某个模型转换失败，提示某个Paddle OP不支持时，可以通过以下方式查找对应的Paddle API：  
1.通常情况下Paddle的OP和API名字接近，我们可以通过[Paddle文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/index_cn.html)直接查询Paddle OP对应的Paddle API。  
2.如果通过Paddle OP名无法搜到或者不确定时需要到[Paddle源码](https://github.com/PaddlePaddle/Paddle)中去搜索。  
- 首先下载Paddle源码，然后到Paddle/python/paddle文件夹下全局搜索OP名，如果搜索到某个OP的python接口使用到了该OP，则表示该API为对应结果。
- 由于paddle.fluid中的API不推荐使用，所以避免找其中的API。  
- 比如需要查找size这个Paddle OP对应的API，在Paddle文档中并不能找到结果，在Paddle/python/paddle文件夹下全局搜索到Paddle/python/paddle/tensor/stat.py脚本中paddle.numel API接口使用到了size op，因此可确认paddle.numel为size op的对应API。
![图片](../imgs/numel.png)

3.找到对应API后需要到Paddle文档中掌握其使用方法，尤其是其输入和属性，在ONNX OP实现Paddle OP时需要尽可能将所有功能都实现。
### ONNX OP实现Paddle OP
掌握Paddle OP的原理和使用方式后，查阅[ONNX OP列表](https://github.com/onnx/onnx/blob/master/docs/Operators.md)找到对应的实现，若ONNX OP和Paddle OP没有一对一的实现，则需要根据Paddle OP的原理使用多个ONNX OP组合实现。  
新的OP转换请在Paddle2ONNX/paddle2onnx/mapper文件夹中实现，并根据API的类别实现于相应的文件夹或者文件中，具体步骤如下：  
1. 若OP在Paddle中属于nn类，比如conv2d，在Paddle中位于paddle.nn下，因此将conv2d的实现放在nn文件夹下，再如clip位于paddle.tensor下，因此将clip的实现放在tensor文件夹下。
2. 每个OP的实现需要有op_name.h和op_name.cc两个文件，将类定义和实现分别实现于不同文件中。
3. 在op_name.h需要实现类的定义，在该文件中需要实现以下几个函数。
> **OPMapper构造函数**：如果有OP中必出现的attr，在该函数中取值。  
> **GetMinOpset**：该函数会在模型转换之前运行，返回该OP支持的最小opset version，所有的OP要求实现opset version 7～15，在该函数中要求获取到确切的可转换的版本，在返回具体的opset version之后不允许报错，如果OP默认1～15都可转换，可以不实现该函数，默认最小opset version 7，如果在某些情况下转换无法进行，请返回-1，否则转换一定要能正确进行。  
> **Opsetn**：其中n表示opset version的版本号，在该函数中实现具体的OP转换，在该函数中要确保转换能够正确进行。  
> **REGISTER_MAPPER**：REGISTER_MAPPER(opname, OpnameMapper)，op的注册函数，调用该宏将op转换注册到Paddle2ONNX中。  

下面以Paddle的gelu OP为例，gelu为一个激活函数，所以直接实现于activation文件中。  
```
// 类的定义
class GeluMapper : public Mapper {
 public:
  GeluMapper(const PaddleParser& p, int64_t block_id, int64_t op_id)
      : Mapper(p, block_id, op_id) {}
  int32_t GetMinOpset(bool verbose = false) { return 9; }
  void Opset9(OnnxHelper* helper);
};

// 类的实现
REGISTER_MAPPER(gelu, GeluMapper)
void GeluMapper::Opset9(OnnxHelper* helper) {
  std::vector<TensorInfo> input_info =
      parser_->GetOpInput(block_idx_, op_idx_, "X");
  std::vector<TensorInfo> output_info =
      parser_->GetOpOutput(block_idx_, op_idx_, "Out");
  auto input_onnx_dtype = GetOnnxDtype(input_info[0].dtype);
  auto op = parser_->GetOpDesc(block_idx_, op_idx_);
  double sqrt_2_value = 1.4142135623730951;
  double scale_value = 0.5;
  double const_1_value = 1.0;
  auto sqrt_2 = helper->MakeConstant({1}, ONNX_NAMESPACE::TensorProto::FLOAT, sqrt_2_value);
  auto scale = helper->MakeConstant({1}, ONNX_NAMESPACE::TensorProto::FLOAT, scale_value);
  auto const_1 = helper->MakeConstant({1}, ONNX_NAMESPACE::TensorProto::FLOAT, const_1_value);
  auto input_name = helper->AutoCast(input_info[0].name, input_info[0].dtype, P2ODataType::FP32);
  // the computation formula follows
  // https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/gelu_cn.html#gelu
  auto erf0 = helper->MakeNode("Div", {input_name, sqrt_2->output(0)});
  auto erf1 = helper->MakeNode("Erf", {erf0->output(0)});
  auto gelu0 = helper->MakeNode("Add", {erf1->output(0), const_1->output(0)});
  auto gelu1 = helper->MakeNode("Mul", {input_name, gelu0->output(0)});
  if (input_info[0].dtype != P2ODataType::FP32) {
    auto out = helper->MakeNode("Mul", {gelu1->output(0), scale->output(0)});
    auto cast_out = helper->MakeNode("Cast", {out->output(0)}, {output_info[0].name});
    AddAttribute(cast_out, "to", GetOnnxDtype(input_info[0].dtype));
  } else {
    helper->MakeNode("Mul", {gelu1->output(0), scale->output(0)}, {output_info[0].name});
  }
}
```

4. 注册多版本的算子转换：实现多个Opsetn函数，后面的数字n表明支持转换到的opset version下限，例如Opset7意味着用户指定转出opset version>=n的情况下，都会选择opset7这个方法来实现具体的转换逻辑，但如果同时实现了Opset9，且用户指定转出opset version>=9时，会优先选择Opset9的方法来转换。
5. 单个文件中实现多个OP请参考：[ActivationMapper](https://github.com/PaddlePaddle/Paddle2ONNX/blob/cpp/paddle2onnx/mapper/activation.h)

#### CPP ONNX组网说明
```
helper->MakeNode接口
参数：
    std::string op_type, # onnx算子的type
    std::vector<std::string> inputs, # 该算子的输入
    std::vector<std::string> outputs, # 该算子输出
return
std::shared_ptr<ONNX_NAMESPACE::NodeProto>node # 算子的输出

AddAttribute接口
参数：
    std::shared_ptr<ONNX_NAMESPACE::NodeProto> node, # 需要添加属性的node
    const std::string& name, # 添加的属性名
    const T& value # 添加的属性
```
 - API更多接口请参考：[API接口](https://github.com/PaddlePaddle/Paddle2ONNX/blob/cpp/paddle2onnx/mapper/onnx_helper.h)
 - 输出绑定：请根据Paddle node的输出名称，为组网结束时的OP指定outputs。
 - 实现OP转换时请将不同opset version版本的OP都实现。  

**注意**：具体实现请多参考Paddle2ONNX python版本的实现，如果有接口不清楚可以多参考已经实现的OP，都有使用示例。

### 实现Paddle OP转换的单测
为了确保转换的正确性，请在OP实现完成之后为该转换写单测。
一个简单的例子如下：
```
from auto_scan_test import OPConvertAutoScanTest, BaseNet
from hypothesis import reproduce_failure
import hypothesis.strategies as st
import numpy as np
import unittest
import paddle

class Net(BaseNet):
    """
    simple Net
    """

    def forward(self, inputs):
        """
        forward
        """
        x = paddle.nn.functional.elu(inputs, alpha=self.config["alpha"])
        return x

class TestEluConvert(OPConvertAutoScanTest):
    """
    api: paddle.nn.functional.elu
    OPset version: 7, 9, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=10, max_value=20), min_size=4, max_size=4))

        alpha = draw(st.floats(min_value=1.0, max_value=10.0))

        dtype = draw(st.sampled_from(["float32"]))

        config = {
            "op_names": ["elu"],
            "test_data_shapes": [input_shape],
            "test_data_types": [[dtype]],
            "opset_version": [7, 9, 15],
            "input_spec_shape": [],
            "alpha": alpha
        }

        models = Net(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)

if __name__ == "__main__":
    unittest.main()
```
 一个单测需要实现的类和函数如下：
1. 一个组网类，继承自BaseNet，只需要写forward函数便可，所有的参数都可以从self.config中获取。
2. 单测类，继承自OPConvertAutoScanTest，需要写sample_convert_config和test两个函数。
#### 组网类
1. 继承自BaseNet，不需写__init__，只需实现forward便可。
1. 将config传入到Net中，然后在self.config中取出你所有想要的数据。
```
class Net(BaseNet):
    def forward(self, inputs, weight):
        x = paddle.nn.functional.conv2d(
            inputs,
            weight,
            stride=self.config["stride"], #从self.config中取数据
            padding=self.config["padding"],
            dilation=self.config["dilation"],
            groups=self.config["groups"],
            data_format=self.config["data_format"])
        return x
```

#### 单测类
1. 单测类继承自OPConvertAutoScanTest。
2. sample_convert_config函数首先根据测试API的文档随机生成所有可测的数值，然后将所有需要用到的数据放到config中，config是一个dict，需传入到组网类中，sample_convert_config函数的返回值为( config, model )。
3. sample_convert_config函数中的config注意必须包括以下key：
> **op_names**：`list of str`，需要检查的OP名，如：["conv2d"]表示要测试的OP为conv2d。  
> **test_data_shapes**：`list of list`，测试数据的shape，如：[[10, 32, 10, 10], [64, 32, 3, 3]]表示第一个输入的shape为[10, 32, 10, 10]，第二个输入的shape为[64, 32, 3, 3]。  
> **test_data_types**：`list of list`，测试数据的数据类型，长度必须和`test_data_shapes`一致，如：[[“float32“, "float64"], ["int32",  "int64"]]表示第一个输入支持的数据类型为“float32“和"float64"，第二个输入支持的数据类型为"int32"和"int64"。  
> **opset_version**：`list`，表示需要测试的opset version，必须包括15，如[9]表示测试opset version为9的转换，此处的设置需要根据op_mapper的实现来设置。  
> **input_spec_shape**：`list of list`，为了支持动态shape而设置，如[[-1, 3, -1, -1],[-1, 3, -1, -1]]表示两个输入都为动态shape，如果不需要测试动态shape的转换，请支持设置为[]。  
4.其他所有的参数都可以放到config中，然后在Net中取出需要的数据，同时config中的数据在运行单测时也会实时打印出来便于调试。
5.返回参数`model`是一个Net()对象或者list of Net()，list of Net()可以实现一个单测测试多个OP转换，具体可参考[`test_auto_scan_unary_ops.py`](https://github.com/PaddlePaddle/Paddle2ONNX/blob/develop/tests/test_auto_scan_unary_ops.py)

> 单个单测测试单个API示例：[`test_auto_scan_conv2d.py`](https://github.com/PaddlePaddle/Paddle2ONNX/blob/develop/tests/test_auto_scan_conv2d.py)  
> 单个单测测试多个API示例：[`test_auto_scan_unary_ops.py`](https://github.com/PaddlePaddle/Paddle2ONNX/blob/develop/tests/test_auto_scan_unary_ops.py)  
> 支持生成自定义数据，请参考：[`test_auto_scan_lookup_table_v2.py`](https://github.com/PaddlePaddle/Paddle2ONNX/blob/develop/tests/test_auto_scan_lookup_table_v2.py)  
> **注意**：所有输入、属性和数据类型都要测试完整。
```
op_api_map = {
    "relu": paddle.nn.functional.relu,
    "sigmoid": paddle.nn.functional.sigmoid
}


class Net(BaseNet):
    def forward(self, inputs):
        return op_api_map[self.config["op_names"]](inputs) #根据config选择具体的API进行测试

class TestUnaryOPConvert(OPConvertAutoScanTest):
    """Testcases for all the unary operators."""

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=2, max_value=20), min_size=4, max_size=4))

        data_shapes = input_shape
        input_specs = [-1, input_shape[1], -1, -1]
        config = {
            "op_names": "",
            "test_data_shapes": [data_shapes],
            "test_data_types": [['float32']],
            "opset_version": [9],
            "input_spec_shape": [input_specs],
        }
        models = list() #需要测试的models
        op_names = list()
        for op_name, i in op_api_map.items():
            config["op_names"] = op_name
            models.append(Net(config)) #构造所有的model
            op_names.append(op_name) #每一个model对应要检查的OP
        config["op_names"] = op_names
        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=40)
```

## 为Paddle2ONNX提PR ##
繁荣的生态需要大家的携手共建，期待和感谢大家为PaddlePaddle贡献自己的力量。
为Paddle2ONNX提PR需要的步骤有：
 1. 进入[Paddle2ONNX官方Repo](https://github.com/PaddlePaddle/Paddle2ONNX)，点击右上角的Star关注Repo的最新动向，然后点击Fork将代码克隆到自己的代码库中。
 2. 返回自己的主页，使用git clone将Fork的代码克隆到本地，然后在克隆代码的根目录下运行pre-commit install安装pre-commit钩子，以在提交代码时完成代码风格的检查。
 3. 按照要求进行开发，开发中请依次完成OP转换和单测，并多写英文注释，便于代码更容易让人理解。
 4. 开发完成后将本地修改git commit到本地仓库，然后git push origin XXX到远端仓库，此时回到github中Fork的Repo可以看到为如下提示：
 ![图片](../imgs/creat_pr.png)
 点击 compare&pull request 按钮，然后出现如下界面，这里需要写言简意赅的标题和详细的修改内容。认真填写完成之后点击creat pull request完成PR。
 ![图片](../imgs/open_pr.png)
 5. 进入到Paddle2ONNX的官方Repo可以在[Pull Request](https://github.com/PaddlePaddle/Paddle2ONNX/pulls)中可以看到提交的PR，PR中有CI测试，如果CI测试没有通过，请点击没有通过CI后的Details查看详情并修改，通过CI之后会有专人进行code review和merge。
![图片](../imgs/pr_details.png)
 6. 更详细的资料请参考[Paddle的PR指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/10_contribution/submit_pr_guide_cn.html)
