import paddle
import unittest
import paddle2onnx

class SimpleNet(paddle.nn.Layer):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.fc = paddle.nn.Linear(10, 10)

    def forward(self, x):
        return self.fc(x)


class TestDygraph2OnnxAPI(unittest.TestCase):
    def test_api(self):
        net = SimpleNet()
        input_spec = [paddle.static.InputSpec(shape=[None, 10], dtype='float32')]
        paddle2onnx.dygraph2onnx(net, "simple_net.onnx", input_spec)

if __name__ == '__main__':
    unittest.main()
