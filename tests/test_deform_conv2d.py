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
    op version: 19
    """

    # [TODO] onnxruntime does not fully support DeformConv, by referring to 
    # https://github.com/onnx/onnx/issues/5451#issuecomment-1658439524
    
    op = Net()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    kh, kw = 3, 3
    input = paddle.rand((8, 1, 28, 28))
    offset = paddle.rand((8, 2 * kh * kw, 26, 26))
    mask = paddle.rand((8, kh * kw, 26, 26))
    weight = paddle.rand((16, 1, kh, kw))
    # obj = APIOnnx(op, "deform_conv2d", [19])
    # obj.set_input_data("input_data", input, offset, weight, mask)
    # obj.run()


if __name__ == "__main__":
    test_deform_conv2d()
