import paddle
from onnxbase import APIOnnx


class Net(paddle.nn.Layer):
    """
    simple Net
    """

    def __init__(self):
        super(Net, self).__init__()

    def forward(self, inputs, axis=1):
        """
        forward
        """
        x = paddle.unbind(inputs, axis)
        print(x)
        return x


def test_unbind():
    """
    api: paddle.unbind
    op version: 7
    """
    op = Net()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, "unbind", [7])
    input_data = paddle.to_tensor(
        [[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[11, 22, 33], [44, 55, 66], [77, 88, 99]]]
    ).astype("float32")

    print(input_data)
    obj.set_input_data("input_data", input_data)
    obj.run()
