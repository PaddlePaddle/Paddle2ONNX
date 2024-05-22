import paddle
from onnxbase import APIOnnx, randtool


class Net(paddle.nn.Layer):
    """
    simple Net
    """

    def __init__(self):
        super(Net, self).__init__()

    def forward(self, inputs, offset, weight, mask):
        """
        forward
        """
        x = paddle.vision.ops.deform_conv2d(inputs, offset, weight, mask=mask)
        return x


def test_deform_conv2d():
    """
    api: paddle.vision.ops.deform_conv2d
    op version: 7
    """
    op = Net()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, "deform_conv2d", [7])
    kh, kw = 3, 3
    input = paddle.rand((8, 1, 28, 28))
    offset = paddle.rand((8, 2 * kh * kw, 26, 26))
    mask = paddle.rand((8, kh * kw, 26, 26))
    weight = paddle.rand((16, 1, kh, kw))
    obj.set_input_data("input_data", input, offset, weight, mask)
    obj.run()


"""
测例报错
/root/miniconda3/envs/paddle_onnx/lib/python3.9/site-packages/paddle/static/io.py:610: UserWarning: no variable in your model, please ensure there are any variables in your model to save
  warnings.warn(
[Paddle2ONNX] Start to parse PaddlePaddle model...
[Paddle2ONNX] Model file path: deform_conv2d/cliped_model.pdmodel
[Paddle2ONNX] Parameters file path: 
[Paddle2ONNX] Start to parsing Paddle model...
[Paddle2ONNX] Use opset_version = 7 for ONNX export.
[Paddle2ONNX] The exported ONNX model is invalid.
[Paddle2ONNX] Model checker error log: No Op registered for DeformConv with domain_version of 7

==> Context: Bad node spec for node. Name: p2o.DeformConv.0 OpType: DeformConv
[Paddle2ONNX] PaddlePaddle model is exported as ONNX format now.
Traceback (most recent call last):
  File "/wuzp/Paddle2ONNX/tests/test_deform_conv2d.py", line 40, in <module>
    test_deform_conv2d()
  File "/wuzp/Paddle2ONNX/tests/test_deform_conv2d.py", line 36, in test_deform_conv2d
    obj.run()
  File "/wuzp/Paddle2ONNX/tests/onnxbase.py", line 420, in run
    res_fict[str(v)] = self._mk_onnx_res(ver=v)
  File "/wuzp/Paddle2ONNX/tests/onnxbase.py", line 300, in _mk_onnx_res
    sess = InferenceSession(
  File "/root/miniconda3/envs/paddle_onnx/lib/python3.9/site-packages/onnxruntime/capi/onnxruntime_inference_collection.py", line 419, in __init__
    self._create_inference_session(providers, provider_options, disabled_optimizers)
  File "/root/miniconda3/envs/paddle_onnx/lib/python3.9/site-packages/onnxruntime/capi/onnxruntime_inference_collection.py", line 472, in _create_inference_session
    sess = C.InferenceSession(session_options, self._model_path, True, self._read_config_from_model)
onnxruntime.capi.onnxruntime_pybind11_state.InvalidGraph: [ONNXRuntimeError] : 10 : INVALID_GRAPH : Load model from /wuzp/Paddle2ONNX/deform_conv2d/deform_conv2d_7.onnx failed:This is an invalid model. In Node, ("p2o.DeformConv.0", DeformConv, "", -1) : ("0": tensor(float),"2": tensor(float),"1": tensor(float),) -> ("save_infer_model/scale_0.tmp_0": tensor(float),) , Error No Op registered for DeformConv with domain_version of 7
"""

if __name__ == "__main__":
    test_deform_conv2d()
