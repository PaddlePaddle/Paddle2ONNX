# 用于配置ONNX模型，方便复用代码
import onnxruntime as rt


class ONNXConfig:
    def __init__(self, onnx_model_path=None, need_simplify=False):
        print("****************************** ONNXConfig ******************************")
        # 判断基本条件是否被满足
        assert (onnx_model_path is not None), "onnx_model_path is empty"

        # 读取模型
        self.onnx_model_path = onnx_model_path

        # 是否需要进行简化模型
        if need_simplify:
            print("-> onnx need simplify")
            print("->-> start simplifying onnx")
            self.simplify()
            print("->-> simplified model complete")
            print("->-> read new onnx model")
        else:
            print("-> onnx don't need simplify")

        # 获取模型的输入和输出
        self.sess = rt.InferenceSession(onnx_model_path)
        self.input_name = [input_name.name for input_name in self.sess.get_inputs()]
        self.input_shape = [input_name.shape for input_name in self.sess.get_inputs()]
        self.output_name = [output_name.name for output_name in self.sess.get_outputs()]
        print("-> onnx path is", self.onnx_model_path)
        print("-> onnx input_name is", self.input_name)
        print("-> onnx input_shape is", self.input_shape)
        print("-> onnx output_name is", self.output_name)
        print("****************************** ONNXConfig ******************************")

    def infer(self, data):
        # 推理
        results = self.sess.run(self.output_name, {self.input_name[0]: data})
        return results

    def simplify(self):
        # 简化模型
        import onnx
        import onnxsim
        import os
        save_name = os.path.basename(self.onnx_model_path).split(".")[0] + "_sim.onnx"
        save_base_path = os.path.dirname(self.onnx_model_path)
        save_path = os.path.join(save_base_path, save_name)
        onnx_model = onnx.load(self.onnx_model_path)
        model_sim, check = onnxsim.simplify(onnx_model)
        onnx.save(model_sim, save_path)
        self.onnx_model_path = save_path
