# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

import numpy as np
import logging
from paddle2onnx.constant import dtypes
from paddle2onnx.op_mapper import OpMapper as op_mapper


@op_mapper(['multiclass_nms', 'multiclass_nms2'])
class MultiClassNMS():
    support_opset_verision_range = (10, 12)
    """
    Convert the paddle multiclass_nms to onnx op.
    This op is get the select boxes from origin boxes.
    """

    @classmethod
    def opset_10(cls, graph, node, **kw):
        """
        Opset an opset of the graph

        Args:
            cls: (todo): write your description
            graph: (todo): write your description
            node: (todo): write your description
            kw: (todo): write your description
        """
        result_name = node.output('Out', 0)
        background = node.attr('background_label')
        normalized = node.attr('normalized')
        if normalized == False:
            logging.warn(
                        "The parameter normalized of multiclass_nms OP of Paddle is False, which has diff with ONNX." \
                        " Please set normalized=True in multiclass_nms of Paddle, see doc Q1 in" \
                        " https://github.com/PaddlePaddle/paddle2onnx/blob/develop/FAQ.md")

        #convert the paddle attribute to onnx tensor
        node_score_threshold = graph.make_node(
            'Constant',
            inputs=[],
            dtype=dtypes.ONNX.FLOAT,
            value=[float(node.attr('score_threshold'))])

        node_iou_threshold = graph.make_node(
            'Constant',
            inputs=[],
            dtype=dtypes.ONNX.FLOAT,
            value=[float(node.attr('nms_threshold'))])

        node_keep_top_k = graph.make_node(
            'Constant',
            inputs=[],
            dtype=dtypes.ONNX.INT64,
            value=[np.int64(node.attr('keep_top_k'))])

        node_keep_top_k_2D = graph.make_node(
            'Constant',
            inputs=[],
            dtype=dtypes.ONNX.INT64,
            dims=[1, 1],
            value=[node.attr('keep_top_k')])

        # the paddle data format is x1,y1,x2,y2
        kwargs = {'center_point_box': 0}

        node_select_nms= graph.make_node(
            'NonMaxSuppression',
            inputs=[node.input('BBoxes', 0), node.input('Scores', 0), node_keep_top_k,\
                node_iou_threshold, node_score_threshold])
        # step 1 nodes select the nms class

        # create some const value to use
        node_const_value = [result_name+"@const_0",
            result_name+"@const_1",\
            result_name+"@const_2",\
            result_name+"@const_-1"]
        value_const_value = [0, 1, 2, -1]
        for name, value in zip(node_const_value, value_const_value):
            graph.make_node(
                'Constant',
                layer_name=name,
                inputs=[],
                outputs=[name],
                dtype=dtypes.ONNX.INT64,
                value=[value])

        # In this code block, we will deocde the raw score data, reshape N * C * M to 1 * N*C*M
        # and the same time, decode the select indices to 1 * D, gather the select_indices
        node_gather_1 = graph.make_node(
            'Gather',
            inputs=[node_select_nms, result_name + "@const_1"],
            axis=1)

        node_squeeze_gather_1 = graph.make_node(
            'Squeeze', inputs=[node_gather_1], axes=[1])

        node_gather_2 = graph.make_node(
            'Gather',
            inputs=[node_select_nms, result_name + "@const_2"],
            axis=1)

        #slice the class is not 0
        if background == 0:
            node_nonzero = graph.make_node(
                'NonZero', inputs=[node_squeeze_gather_1])
        else:
            node_thresh = graph.make_node(
                'Constant', inputs=[], dtype=dtypes.ONNX.INT32, value=[-1])

            node_cast = graph.make_node(
                'Cast', inputs=[node_squeeze_gather_1], to=6)

            node_greater = graph.make_node(
                'Greater', inputs=[node_cast, node_thresh])

            node_nonzero = graph.make_node('NonZero', inputs=[node_greater])

        node_gather_1_nonzero = graph.make_node(
            'Gather', inputs=[node_gather_1, node_nonzero], axis=0)

        node_gather_2_nonzero = graph.make_node(
            'Gather', inputs=[node_gather_2, node_nonzero], axis=0)

        # reshape scores N * C * M to (N*C*M) * 1
        node_reshape_scores_rank1 = graph.make_node(
            "Reshape",
            inputs=[node.input('Scores', 0), result_name + "@const_-1"])

        # get the shape of scores
        node_shape_scores = graph.make_node(
            'Shape', inputs=node.input('Scores'))

        # gather the index: 2 shape of scores
        node_gather_scores_dim1 = graph.make_node(
            'Gather',
            inputs=[node_shape_scores, result_name + "@const_2"],
            axis=0)

        # mul class * M
        node_mul_classnum_boxnum = graph.make_node(
            'Mul', inputs=[node_gather_1_nonzero, node_gather_scores_dim1])

        # add class * M * index
        node_add_class_M_index = graph.make_node(
            'Add', inputs=[node_mul_classnum_boxnum, node_gather_2_nonzero])

        # Squeeze the indices to 1 dim
        node_squeeze_select_index = graph.make_node(
            'Squeeze', inputs=[node_add_class_M_index], axes=[0, 2])

        # gather the data from flatten scores
        node_gather_select_scores = graph.make_node(
            'Gather',
            inputs=[node_reshape_scores_rank1, node_squeeze_select_index],
            axis=0)

        # get nums to input TopK
        node_shape_select_num = graph.make_node(
            'Shape', inputs=[node_gather_select_scores])

        node_gather_select_num = graph.make_node(
            'Gather',
            inputs=[node_shape_select_num, result_name + "@const_0"],
            axis=0)

        node_unsqueeze_select_num = graph.make_node(
            'Unsqueeze', inputs=[node_gather_select_num], axes=[0])

        node_concat_topK_select_num = graph.make_node(
            'Concat',
            inputs=[node_unsqueeze_select_num, node_keep_top_k_2D],
            axis=0)

        node_cast_concat_topK_select_num = graph.make_node(
            'Cast', inputs=[node_concat_topK_select_num], to=6)
        # get min(topK, num_select)
        node_compare_topk_num_select = graph.make_node(
            'ReduceMin', inputs=[node_cast_concat_topK_select_num], keepdims=0)

        # unsqueeze the indices to 1D tensor
        node_unsqueeze_topk_select_indices = graph.make_node(
            'Unsqueeze', inputs=[node_compare_topk_num_select], axes=[0])

        # cast the indices to INT64
        node_cast_topk_indices = graph.make_node(
            'Cast', inputs=[node_unsqueeze_topk_select_indices], to=7)

        # select topk scores  indices
        outputs_topk_select_topk_indices = [result_name + "@topk_select_topk_values",\
            result_name + "@topk_select_topk_indices"]
        node_topk_select_topk_indices = graph.make_node(
            'TopK',
            inputs=[node_gather_select_scores, node_cast_topk_indices],
            outputs=outputs_topk_select_topk_indices)

        # gather topk label, scores, boxes
        node_gather_topk_scores = graph.make_node(
            'Gather',
            inputs=[
                node_gather_select_scores, outputs_topk_select_topk_indices[1]
            ],
            axis=0)

        node_gather_topk_class = graph.make_node(
            'Gather',
            inputs=[
                node_gather_1_nonzero, outputs_topk_select_topk_indices[1]
            ],
            axis=1)

        # gather the boxes need to gather the boxes id, then get boxes
        node_gather_topk_boxes_id = graph.make_node(
            'Gather',
            inputs=[
                node_gather_2_nonzero, outputs_topk_select_topk_indices[1]
            ],
            axis=1)

        # squeeze the gather_topk_boxes_id to 1 dim
        node_squeeze_topk_boxes_id = graph.make_node(
            'Squeeze', inputs=[node_gather_topk_boxes_id], axes=[0, 2])

        node_gather_select_boxes = graph.make_node(
            'Gather',
            inputs=[node.input('BBoxes', 0), node_squeeze_topk_boxes_id],
            axis=1)

        # concat the final result
        # before concat need to cast the class to float
        node_cast_topk_class = graph.make_node(
            'Cast', inputs=[node_gather_topk_class], to=1)

        node_unsqueeze_topk_scores = graph.make_node(
            'Unsqueeze', inputs=[node_gather_topk_scores], axes=[0, 2])

        inputs_concat_final_results = [node_cast_topk_class, node_unsqueeze_topk_scores, \
            node_gather_select_boxes]
        node_sort_by_socre_results = graph.make_node(
            'Concat', inputs=inputs_concat_final_results, axis=2)

        # select topk classes indices
        node_squeeze_cast_topk_class = graph.make_node(
            'Squeeze', inputs=[node_cast_topk_class], axes=[0, 2])
        node_neg_squeeze_cast_topk_class = graph.make_node(
            'Neg', inputs=[node_squeeze_cast_topk_class])

        outputs_topk_select_classes_indices = [result_name + "@topk_select_topk_classes_scores",\
            result_name + "@topk_select_topk_classes_indices"]
        node_topk_select_topk_indices = graph.make_node(
            'TopK',
            inputs=[node_neg_squeeze_cast_topk_class, node_cast_topk_indices],
            outputs=outputs_topk_select_classes_indices)
        node_concat_final_results = graph.make_node(
            'Gather',
            inputs=[
                node_sort_by_socre_results,
                outputs_topk_select_classes_indices[1]
            ],
            axis=1)
        node_concat_final_results = graph.make_node(
            'Squeeze',
            inputs=[node_concat_final_results],
            outputs=[node.output('Out', 0)],
            axes=[0])

        if node.type == 'multiclass_nms2':
            graph.make_node(
                'Squeeze',
                inputs=[node_gather_2_nonzero],
                outputs=node.output('Index'),
                axes=[0])
            #graph.make_node(
            #    'Unsqueeze', 
            #    inputs=[outputs_topk_select_classes_indices[1]],
            #    outputs=node.output('Index'), 
            #    axes=[-1])
