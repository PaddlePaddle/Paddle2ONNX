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
为了确保转换的正确性，OP开发完成后需要进行转换测试，请将Paddle2ONNX develop分支中tests文件夹下OP的单测拷贝到tests文件夹下并测试，并确保单测能够顺利运行，单测文件中的op_names表示该单测测试的OP。
运行单测命令如下：
```
python test_auto_scan_xx.py
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
