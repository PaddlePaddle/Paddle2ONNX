from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import paddle2onnx.paddle2onnx_cpp2py_export as c_p2o
from paddle2onnx import version
import onnx

def export(model_filename, params_filename="", save_file=None, opset_version=9, auto_upgrade_opset=True, verbose=True):
    onnx_proto = onnx.ModelProto()
    onnx_proto_str = c_p2o.export(model_filename, params_filename, opset_version, auto_upgrade_opset, verbose)
    onnx_proto.ParseFromString(onnx_proto_str)
    if save_file is not None:
        with open(save_file, "wb") as f:
            f.write(onnx_proto.SerializeToString())

def dygraph2onnx(layer, save_file, input_spec, opset_version=9, **configs):
    import os
    import paddle
    dirname = os.path.split(save_file)[-1]
    paddle.jit.save(layer, os.path.join(dirname, "model"), input_spec)
    export(os.path.join(dirname, 'model.pdmodel'), os.path.join(dirname, 'model.pdiparams'), save_file=save_file, opset_version=opset_version, auto_upgrade_opset=False)
