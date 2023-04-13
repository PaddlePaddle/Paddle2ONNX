# Paddle2ONNX 开发指南
本文档为 Paddle2ONNX 的新 OP 开发指南，旨在帮助开发者快速掌握 Paddle2ONNX 的新 OP 开发方式，在遇到模型转换失败时能有应对方法，快速解决自己遇到的问题。

## Paddle2ONNX 简介 ##
ONNX (Open Neural Network Exchange) 是针对机器学习所设计的开源文件格式，用于存储训练好的模型。它使得不同的人工智能框架可以采用相同格式存储模型并交互。通过 ONNX 格式，Paddle 模型可以使用 OpenVINO、ONNX Runtime 和 TensorRT 等推理框架进行推理。

Paddle2ONNX 是 PaddlePaddle 的核心工具套件之一，负责将 Paddle 的 Inference 模型转换为 ONNX 格式，便于开发者将 Paddle 模型扩展到支持 ONNX 部署的框架上进行推理。

## Paddle2ONNX 新 OP 开发步骤 ##
Paddle2ONNX 开发的主要步骤为：  
1. 根据 Paddle OP 查阅对应的 Paddle API 并掌握其使用方法；  
2. 根据 Paddle OP 的原理通过 ONNX OP 直接或者组合实现相同功能；  
3. 为 Paddle OP 的转换添加单测；  
4. 为 Paddle2ONNX 提 PR；  

### 查找 Paddle API 和 paddle OP 对应关系 ###
Paddle OP 的转换需要掌握 Paddle OP 的原理和使用方式，因此需要查阅 Paddle OP 对应的 Paddle API。  
当遇到某个模型转换失败，提示某个 Paddle OP 不支持时，可以通过以下方式查找对应的 Paddle API：  
1. 通常情况下 Paddle 的 OP 和 API 名字接近，我们可以通过[Paddle文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/index_cn.html)直接查询 Paddle OP 对应的 Paddle API。

2. 如果通过 Paddle OP 名无法搜到或者不确定时需要到[Paddle源码](https://github.com/PaddlePaddle/Paddle)中去搜索。  
- 首先下载 Paddle 源码，然后到 Paddle/python/paddle 文件夹下全局搜索 OP 名，如果搜索到某个 OP 的 python 接口使用到了该 OP，则表示该 API 为对应结果。
- 由于 paddle.fluid 中的 API 不推荐使用，所以避免找其中的 API，单测文件中也要避免使用 paddle.fluid 的 API。  
- 例如：需要查找 size OP 对应的 API，在 Paddle 文档中并不能找到结果，在 Paddle/python/paddle 文件夹下全局搜索到 Paddle/python/paddle/tensor/stat.py 脚本中 paddle.numel API 接口使用到了 size op，因此可确认 paddle.numel 为 size op 的对应 API。
![图片](https://user-images.githubusercontent.com/30516196/231719526-101e2e27-dfeb-4003-a1be-96297d70ed8c.png)

3. 找到对应 API 后需要到 Paddle 文档中掌握其使用方法，尤其是其输入和属性，在 ONNX OP 实现 Paddle OP 时需要尽可能将所有功能都实现，如果不能实现的属性需要给出相应的提示。size OP 对应的 输入输出和属性等需要在 paddle/fluid/operators 下搜索 SizeOpMaker，获得的结果便是 Size OP对应的输入输出属性等所有信息。
![图片](https://user-images.githubusercontent.com/30516196/231705787-169ed5fc-494e-4404-a2c3-fa461e7cb962.png)

### ONNX OP 实现 Paddle OP相同的功能
掌握Paddle OP的原理和使用方式后，查阅[ONNX OP列表](https://github.com/onnx/onnx/blob/master/docs/Operators.md)找到对应的实现，若ONNX OP和Paddle OP没有一对一的实现，则需要根据Paddle OP的原理使用多个ONNX OP组合实现。  

下面以一个稍微复杂的算子 Paddle 的 argsort OP 为例，首先按照上述步骤查找 argsort 对应的 API 和原理，以及其所有的输入和输出及属性等信息。

新的 OP 转换实现需要根据算子的类别确认：
1. 如是 conv2d、pool2d 和 pad 等算子，需要将算子转换实现于 Paddle2ONNX/paddle2onnx/mapper/nn 中
2. 如是 rule、relu6 和 sigmod 等激活值类别算子，需要将算子转换实现于 Paddle2ONNX/paddle2onnx/mapper/activation.* 中
3. 如是 eye、cumsum 和 index_sample 等 tensor 处理类算子，需要将算子转换实现于 Paddle2ONNX/paddle2onnx/mapper/tensor 中

argsort 需要在 Paddle2ONNX/paddle2onnx/mapper/tensor 文件夹中实现，首先根据 OP 名称新生成 argsort.h 和 argsort.cc 然后转换实现于相应的文件中：

在头文件中实现 OPNameMapper 类，继承 Mapper 基类。
```
class ArgsortMapper : public Mapper {
 public:
  ArgsortMapper(const PaddleParser& p, OnnxHelper* helper, int64_t block_id,
                int64_t op_id)
      : Mapper(p, helper, block_id, op_id) {
    GetAttr("descending", &descending_);
    GetAttr("axis", &axis_);
  }
  int32_t GetMinOpset(bool verbose = false);
  void Opset10();
  void Opset7();

 private:
  bool descending_;
  int64_t axis_;
};
```

 - 在构造函数中将 OP 的所有属性获取，并存到私有成员变量中
 - Paddle2ONNX 需要实现 Opset version 7～16，如果实现的 OP 不是从 Opset version 7 开始或者由于 Paddle OP中的某些属性导致无法导出为ONNX，则需要重写 GetMinOpset 函数，该函数返回 -1 表示该 OP 无法导出为 ONNX，否则表示 导出该 OP 所需的最小 Opset version
 - OpsetX 表示 opset version 为 x 时的转换实现，如果定义了 Opset7 和 Opset10 两个转换方法，意味着用户指定转出 opset version 7～9 时，使用 Opset7 中的转换逻辑实现转换，用户指定 opset version 10～16 时，使用 Opset10 中的转换逻辑实现转换。

在源文件中实现具体的逻辑
```
REGISTER_MAPPER(argsort, ArgsortMapper)

int32_t ArgsortMapper::GetMinOpset(bool verbose) {
  if (!descending_) {
    Logger(verbose, 11) << "While descending=False, " << RequireOpset(11)
                        << std::endl;
    return 11;
  }

  if (axis_ < 0) {
    axis_ = axis_ + GetInput("X")[0].Rank();
  }
  if (GetInput("X")[0].shape[axis_] <= 0) {
    Logger(verbose, 10) << "While input shape is dynamic, " << RequireOpset(10)
                        << std::endl;
    return 10;
  }
  return 7;
}

void ArgsortMapper::Opset10() {
  auto x_info = GetInput("X");
  auto output_info = GetOutput("Out");
  auto indices_info = GetOutput("Indices");

  auto shape = helper_->MakeNode("Shape", {x_info[0].name})->output(0);
  if (axis_ < 0) {
    axis_ = axis_ + x_info[0].Rank();
  }
  auto dim_size = helper_->Slice(shape, {0}, {axis_}, {axis_ + 1});

  auto out_node =
      helper_->MakeNode("TopK", {x_info[0].name, dim_size},
                        {output_info[0].name, indices_info[0].name});
  AddAttribute(out_node, "axis", axis_);
  if (helper_->GetOpsetVersion() > 10) {
    if (!descending_) {
      AddAttribute(out_node, "largest", static_cast<int64_t>(0));
    } else {
      AddAttribute(out_node, "largest", static_cast<int64_t>(1));
    }
  }
}

void ArgsortMapper::Opset7() {
  auto x_info = GetInput("X");
  auto output_info = GetOutput("Out");
  auto indices_info = GetOutput("Indices");

  if (axis_ < 0) {
    axis_ = axis_ + x_info[0].Rank();
  }

  auto out_node = helper_->MakeNode(
      "TopK", {x_info[0].name}, {output_info[0].name, indices_info[0].name});
  AddAttribute(out_node, "axis", axis_);
  AddAttribute(out_node, "k", x_info[0].shape[axis_]);
}
```
 - 使用 REGISTER_MAPPER(op, OpMapper) 来注册 OP 的转换逻辑到 Paddle2ONNX
 - GetMinOpset 函数中要将所有的判断逻辑写清楚，提示信息要清晰
 - ONNX组网使用 helper_->MakeNode 函数实现，函数的具体定义可参考：Paddle2ONNX/paddle2onnx/mapper/onnx_helper.cc
 - AddAttribute 可为 Node 新增属性信息，具体的函数定义可参考：Paddle2ONNX/paddle2onnx/mapper/onnx_helper.cc
 - Paddle2ONNX/paddle2onnx/mapper/onnx_helper.h 中提供了大量的辅助组网信息，比如 Split, Transpose, Slice, Reshape 等等，可以有效的简化转换
 - 实现 OP 转换时请将 opset version 7~16 都实现
 - ONNX 的 OP 定义以及 opset version 等信息可以查阅：[ONNX OP 文档](https://github.com/onnx/onnx/blob/main/docs/Operators.md)

**Note** Paddle2ONNX 已经实现了大量的 OP 转换，可以参考 Paddle2ONNX/paddle2onnx/mapper/* 中已经实现的转换来写需要新增的 OP 转换。

### 实现 Paddle OP 转换的单测
为了确保转换的正确性，请在 OP 实现完成之后为该转换写单测。

argsort 的单测示例如下：
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

    def forward(self, input):
        """
        forward
        """

        x = paddle.argsort(
            input,
            axis=self.config['axis'],
            descending=self.config['descending'])
        return x


class TestArgsortConvert(OPConvertAutoScanTest):
    """
    api: paddle.argsort
    OPset version: 11, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=2, max_value=5), min_size=2, max_size=5))

        axis = draw(
            st.integers(
                min_value=-len(input_shape), max_value=len(input_shape) - 1))

        dtype = draw(st.sampled_from(["float32", "float64"]))
        descending = draw(st.booleans())

        def generator_data():
            import random
            import numpy as np
            t = 1
            for i in range(len(input_shape)):
                t = t * input_shape[i]
            input_data = np.array(random.sample(range(-5000, 5000), t))
            input_data = input_data.reshape(input_shape)
            return input_data

        if descending:
            opset_version = [7, 10, 11, 15]
        else:
            opset_version = [11, 15]
        config = {
            "op_names": ["argsort"],
            "test_data_shapes": [generator_data],
            "test_data_types": [[dtype]],
            "opset_version": opset_version,
            "input_spec_shape": [],
            "axis": axis,
            "descending": descending,
        }

        models = Net(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)


if __name__ == "__main__":
    unittest.main()

```

一个单测需要实现的类和函数如下：
1. 一个组网类，继承自 BaseNet，只需要写 forward 函数便可，所有的参数都可以从 self.config 中获取
2. 单测类，继承自 OPConvertAutoScanTest，需要写 sample_convert_config 和 test 两个函数。

#### 组网类
1. 继承自 BaseNet，不需写 __init__，只需实现 forward 便可
2. 将 config 传入到 Net 中，然后在 self.config 中取出你所有想要的数据
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
1. 单测类继承自 OPConvertAutoScanTest。
2. sample_convert_config 函数首先根据测试 API 的文档随机生成所有可测的数值，然后将所有需要用到的数据放到 config 中，config 是一个 dict，需传入到组网类中，sample_convert_config 函数的返回值为 (config, model)
3. sample_convert_config 函数中的 config 注意必须包括以下 key：
> **op_names**：`list of str`，需要检查的 OP 名，如：["conv2d"]表示要测试的 OP 为 conv2d。  
> **test_data_shapes**：`list of list`，测试数据的 shape，如：[[10, 32, 10, 10], [64, 32, 3, 3]]表示第一个输入的 shape 为 [10, 32, 10, 10]，第二个输入的 shape 为 [64, 32, 3, 3]。  
> **test_data_types**：`list of list`，测试数据的数据类型，长度必须和 `test_data_shapes` 一致，如：[[“float32“, "float64"], ["int32",  "int64"]]表示第一个输入支持的数据类型为 “float32“ 和 "float64"，第二个输入支持的数据类型为 "int32" 和 "int64"。  
> **opset_version**：`list`，表示需要测试的 opset version，只需要设置支持的最小 opset version 便可，如 [9] 表示测试opset version为 9～16 的转换。  
> **input_spec_shape**：`list of list`，为了支持动态shape而设置，如 [[-1, 3, -1, -1],[-1, 3, -1, -1]] 表示两个输入都为动态 shape，如果不需要测试动态 shape 的转换，请直接设置为 []。  
4. 其他所有的参数都可以放到 config 中，然后在 Net 中取出需要的数据，同时 config 中的数据在运行单测时也会实时打印出来便于调试。
5. 返回参数 `model` 是一个 Net() 对象或者 list of Net()，list of Net() 可以实现一个单测测试多个 OP 转换，具体可参考[`test_auto_scan_unary_ops.py`](https://github.com/PaddlePaddle/Paddle2ONNX/blob/develop/tests/test_auto_scan_unary_ops.py)

> 单个单测测试单个 API 示例：[`test_auto_scan_conv2d.py`](https://github.com/PaddlePaddle/Paddle2ONNX/blob/develop/tests/test_auto_scan_conv2d.py)  
> 单个单测测试多个 API 示例：[`test_auto_scan_unary_ops.py`](https://github.com/PaddlePaddle/Paddle2ONNX/blob/develop/tests/test_auto_scan_unary_ops.py)  
> 支持生成自定义数据，请参考：[`test_auto_scan_lookup_table_v2.py`](https://github.com/PaddlePaddle/Paddle2ONNX/blob/develop/tests/test_auto_scan_lookup_table_v2.py)  
> **注意**：所有输入、属性和数据类型都要测试完整。

## 为 Paddle2ONNX 提 PR ##
繁荣的生态需要大家的携手共建，期待和感谢大家为 PaddlePaddle 贡献自己的力量

为 Paddle2ONNX 提 PR 需要的步骤有：
 1. 进入[Paddle2ONNX 官方 Repo](https://github.com/PaddlePaddle/Paddle2ONNX)，点击右上角的 Star 关注 Repo 的最新动向，然后点击 Fork 将代码克隆到自己的代码库中。
 2. 返回自己的主页，使用 git clone 将 Fork 的代码克隆到本地，然后在克隆代码的根目录下运行 pre-commit install 安装 pre-commit 钩子，以在提交代码时完成代码风格的检查。
 3. 按照要求进行开发，开发中请依次完成 OP 转换和单测，并多写英文注释，便于代码更容易让人理解。
 4. 开发完成后将本地修改 git commit 到本地仓库，然后 git push origin XXX 到远端仓库，此时回到 github 中 Fork 的 Repo 可以看到为如下提示：
 ![图片](https://user-images.githubusercontent.com/30516196/231718803-d2b3c940-144e-44e3-8b2b-ae93cc01f8f2.png)
 点击 compare&pull request 按钮，然后出现如下界面，这里需要写言简意赅的标题和详细的修改内容。认真填写完成之后点击 creat pull request 完成 PR。
 ![图片](https://user-images.githubusercontent.com/30516196/231719191-7fcf812a-e0b4-4db0-b66e-fade3e5be5bc.png)
 5. 进入到 Paddle2ONNX 的官方 Repo 可以在[Pull Request](https://github.com/PaddlePaddle/Paddle2ONNX/pulls) 中可以看到提交的 PR，PR 中有 CI 测试，如果 CI 测试没有通过，请点击没有通过 CI 后的 Details 查看详情并修改，通过 CI 之后会有专人进行 code review 和 merge。
![图片](https://user-images.githubusercontent.com/30516196/231719335-d61c433f-70f3-4cc1-b7f5-29f2bab7e69a.png)
 6. 更详细的资料请参考[Paddle 的 PR 指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/10_contribution/submit_pr_guide_cn.html)
