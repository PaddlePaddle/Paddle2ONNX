from onnx import onnx_pb2
import paddle.fluid.core as core


PADDLE_TO_ONNX_DTYPE = {
    core.VarDesc.VarType.FP32: onnx_pb2.TensorProto.FLOAT,
    core.VarDesc.VarType.FP64: onnx_pb2.TensorProto.FLOAT16,
    '': onnx_pb2.TensorProto.DOUBLE,
    core.VarDesc.VarType.INT32: onnx_pb2.TensorProto.INT32,
    core.VarDesc.VarType.INT16: onnx_pb2.TensorProto.INT16,
    '': onnx_pb2.TensorProto.INT8,
    '': onnx_pb2.TensorProto.UINT8,
    core.VarDesc.VarType.INT16: onnx_pb2.TensorProto.UINT16,
    core.VarDesc.VarType.INT64: onnx_pb2.TensorProto.INT64,
    '': onnx_pb2.TensorProto.STRING,
    '': onnx_pb2.TensorProto.COMPLEX64,
    '': onnx_pb2.TensorProto.COMPLEX128,
    core.VarDesc.VarType.BOOL: onnx_pb2.TensorProto.BOOL,
}
