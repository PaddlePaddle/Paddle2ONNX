# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import numpy as np
import argparse
from tqdm import tqdm
import paddle
from paddleslim.common import load_config
from paddleslim.common import load_inference_model
from post_process import YOLOPostProcess, coco_metric
from dataset import COCOValDataset
import onnx
from onnx import onnx_pb as onnx_proto
from onnx import helper, TensorProto, ModelProto
import onnxruntime as ort
import itertools
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


def argsparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--config_path',
        type=str,
        default=None,
        help="path of compression strategy config.",
        required=True)
    parser.add_argument(
        '--devices',
        type=str,
        default='gpu',
        help="which device used to compress.")

    return parser


def select_tensors_to_analysis(model):
    value_infos = {vi.name: vi for vi in model.graph.value_info}
    value_infos.update({ot.name: ot for ot in model.graph.output})
    value_infos.update({it.name: it for it in model.graph.input})
    initializer = set(init.name for init in model.graph.initializer)

    tensors_to_analysis = set()
    tensor_type_to_analysis = set([TensorProto.FLOAT, TensorProto.FLOAT16])
    op_types_to_analysis = ["Conv"]

    for node in model.graph.node:
        if len(op_types_to_analysis
               ) == 0 or node.op_type in op_types_to_analysis:
            for tensor_name in itertools.chain(node.input, node.output):
                if tensor_name.count(".b_0") or tensor_name.count(".w_0"):
                    continue
                if tensor_name in value_infos.keys():
                    vi = value_infos[tensor_name]
                    if vi.type.HasField('tensor_type') and (
                            vi.type.tensor_type.elem_type in
                            tensor_type_to_analysis) and (
                                tensor_name not in initializer):
                        tensors_to_analysis.add(tensor_name)

    return tensors_to_analysis, value_infos


def augment_graph(ori_onnx_model):
    model = onnx_proto.ModelProto()
    model.CopyFrom(ori_onnx_model)
    model = onnx.shape_inference.infer_shapes(model)

    tensors, value_infos = select_tensors_to_analysis(model)

    print("tensors len: ", len(tensors))

    added_outputs = []
    for tensor in tensors:
        dim = value_infos[tensor].type.tensor_type.shape.dim
        shape = value_infos[tensor].type.tensor_type.shape
        shape = tuple(dim[i].dim_value for i in range(len(dim)))
        # print("shape: ",shape)
        added_outputs.append(
            onnx.helper.make_tensor_value_info(tensor, TensorProto.FLOAT,
                                               shape))

    model.graph.output.extend(added_outputs)
    onnx.save(model, "augmented_model.onnx")


def collect_ops_histogram(tensor_names, tensor_values):
    histogram_bins = 1000
    hist = {}
    for index in range(len(tensor_names)):
        tensor_name = tensor_names[index].name
        print("total: ",
              len(tensor_names), " index: ", index, " tensor name: ",
              tensor_name)
        var_tensor = np.array(tensor_values[index])
        var_tensor = var_tensor.flatten()
        min_v = float(np.min(var_tensor))
        max_v = float(np.max(var_tensor))
        _, hist_edges = np.histogram(
            var_tensor.copy(), bins=1000, range=(min_v, max_v))
        # print(var_tensor, hist_edges)
        hist[tensor_name] = [var_tensor.copy(), hist_edges.copy()]
    return hist


def draw_pdf(hist_data, save_pdf_name):
    pdf_path = os.path.join(global_config["model_dir"], save_pdf_name)
    with PdfPages(pdf_path) as pdf:
        index = 0
        for name in hist_data:
            index = index + 1
            print("total: ",
                  len(hist_data), " index: ", index, " draw tensor: ", name)
            plt.hist(hist_data[name][0], bins=hist_data[name][1])
            plt.xlabel(name)
            plt.ylabel("frequency")
            plt.title("Hist of variable {}".format(name))
            plt.show()
            pdf.savefig()
            plt.close()
    print('Histogram plot is saved in {}'.format(pdf_path))


def analysis():

    place = paddle.CUDAPlace(0) if FLAGS.devices == 'gpu' else paddle.CPUPlace()

    model_path = global_config["model_dir"] + "/float_model.onnx"
    # model_path = global_config["model_dir"] + "/onnx_quant.onnx"

    model = onnx.load(model_path)
    augment_graph(model)

    model_path = "augmented_model.onnx"
    sess_options = ort.SessionOptions()
    #sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess_options.optimized_model_filepath = global_config[
        "model_dir"] + "/optimize_model.onnx"
    sess = ort.InferenceSession(model_path, sess_options)

    data_iter = 0
    all_outs = None
    for data in val_loader:
        print("infer index: ", data_iter)
        data_all = {k: np.array(v) for k, v in data.items()}
        outs = sess.run(None, {sess.get_inputs()[0].name: data_all['image']})
        if data_iter >= 5:
            break
        elif data_iter == 0:
            all_outs = [np.array(out) for out in outs]
        else:
            for index in range(len(outs)):
                all_outs[index] = np.concatenate(
                    [all_outs[index], np.array(outs[index])], axis=0)
        data_iter = data_iter + 1
    print("start generate hist data ...")
    hist_data = collect_ops_histogram(sess.get_outputs(), all_outs)
    print("finish generate hist data !")
    print("start draw_pdf ...")
    draw_pdf(hist_data, "hist_result.pdf")
    print("finished ! ")


def main():
    global global_config
    all_config = load_config(FLAGS.config_path)
    global_config = all_config["Global"]

    global val_loader
    dataset = COCOValDataset(
        dataset_dir=global_config['dataset_dir'],
        image_dir=global_config['val_image_dir'],
        anno_path=global_config['val_anno_path'])
    global anno_file
    anno_file = dataset.ann_file
    val_loader = paddle.io.DataLoader(dataset, batch_size=1)

    analysis()


if __name__ == '__main__':
    paddle.enable_static()
    parser = argsparser()
    FLAGS = parser.parse_args()

    assert FLAGS.devices in ['cpu', 'gpu', 'xpu', 'npu']
    paddle.set_device(FLAGS.devices)

    main()
