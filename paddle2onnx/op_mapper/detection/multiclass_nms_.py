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
        scores = node.input('Scores', 0)
        bboxes = node.input('BBoxes', 0)
        num_class = node.input_shape('Scores', 0)[1]
        if len(node.input_shape('Scores', 0)) == 2:
            scores = graph.make_node('Transpose', inputs=[scores], perm=[1, 0])
            scores = graph.make_node('Unsqueeze', inputs=[scores], axes=[0])
            bboxes = graph.make_node(
                'Split',
                inputs=[bboxes],
                outputs=num_class,
                axis=1,
                split=[1] * num_class)
            bbox_ids = []
            scores = []
            class_ids = []
            for i, single_class_bboxes in enumerate(bboxes):
                single_class_bboxes = graph.make_node(
                    'Transpose', inputs=[single_class_bboxes], perm=[1, 0, 2])
                class_id, score, bbox_id = cls.lod_nms(graph, node, scores,
                                                       single_class_bboxes, i)
                bbox = graph.make_node('Squeeze', inputs=[out], axes=[0])
                bbox_ids.append(bbox_id)
                scores.append(score)
                class_ids.append(class_id)
            bbox_ids = graph.make_node('Concat', inputs=bbox_ids, axis=0)
            scores = graph.make_node('Concat', inputs=scores, axis=0)
            class_ids = graph.make_node('Concat', inputs=class_ids, axis=0)
            const_shape = graph.make_node(
                'Constant', dtype=dtypes.ONNX.INT64, value=[-1, 4])
            bboxes = graph.make_node('bboxes', inputs=[bboxes, const_shape])
            cls.keep_top_k(graph, node, class_ids, scores, bboxes)
        else:
            cls.nms(graph, node, scores, bboxes)

    @classmethod
    def lod_nms(cls, graph, node, scores, bboxes, class_id=0, **kw):
        background = node.attr('background_label')
        normalized = node.attr('normalized')

        nms_top_k = node.attr('nms_top_k')
        if nms_top_k < 0:
            nms_top_k = 10000
        if normalized == False:
            logging.warn(
                        "The parameter normalized of multiclass_nms OP of Paddle is False, which has diff with ONNX." \
                        " Please set normalized=True in multiclass_nms of Paddle, see doc Q1 in" \
                        " https://github.com/PaddlePaddle/paddle2onnx/blob/develop/FAQ.md")
        #convert the paddle attribute to onnx tensor
        score_threshold = graph.make_node(
            'Constant',
            dtype=dtypes.ONNX.FLOAT,
            value=[float(node.attr('score_threshold'))])

        iou_threshold = graph.make_node(
            'Constant',
            dtype=dtypes.ONNX.FLOAT,
            value=[float(node.attr('nms_threshold'))])

        nms_top_k = graph.make_node(
            'Constant', dtype=dtypes.ONNX.INT64, value=[np.int64(nms_top_k)])

        # the paddle data format is x1,y1,x2,y2
        kwargs = {'center_point_box': 0}

        select_nms = graph.make_node(
            'NonMaxSuppression',
            inputs=[bboxes, scores, nms_top_k, iou_threshold, score_threshold])

        # step 1 nodes select the nms class
        # create some const value to use
        const_values = []
        for value in [0, 1, 2, -1]:
            const_value = graph.make_node(
                dtype=dtypes.ONNX.INT64, value=[value])
            const_values.append(const_value)

        # In this code block, we will deocde the raw score data, reshape N * C * M to 1 * N*C*M
        # and the same time, decode the select indices to 1 * D, gather the select_indices
        class_id = graph.make_node(
            'Gather', inputs=[select_nms, const_values[1]], axis=1)

        class_id = graph.make_node('Squeeze', inputs=[class_id], axes=[1])

        bbox_id = graph.make_node(
            'Gather', inputs=[select_nms, const_values[2]], axis=1)

        #slice the class is not 0
        if background == 0:
            nonzero = graph.make_node('NonZero', inputs=[class_id])

            class_id = graph.make_node(
                'Gather', inputs=[class_id, nonzero], axis=0)

            bbox_id = graph.make_node(
                'Gather', inputs=[bbox_id, nonzero], axis=0)

        # reshape scores N * C * M to (N*C*M) * 1
        scores = graph.make_node('Reshape', inputs=[scores, const_values[-1]])

        # get the shape of scores
        shape_scores = graph.make_node('Shape', inputs=scores)

        # gather the index: 2 shape of scores
        class_num = graph.make_node(
            'Gather', inputs=[shape_scores, const_values[2]], axis=0)

        # mul class * M
        mul_classnum_boxnum = graph.make_node(
            'Mul', inputs=[class_id, class_num])

        # add class * M * index
        add_class_M_index = graph.make_node(
            'Add', inputs=[mul_classnum_boxnum, bbox_id])

        # Squeeze the indices to 1 dim
        score_indices = graph.make_node(
            'Squeeze', inputs=[add_class_M_index], axes=[0, 2])

        # gather the data from flatten scores
        select_scores = graph.make_node(
            'Gather', inputs=[scores, score_indices], axis=0)

        return class_id, scores, bbox_id

    @classmethod
    def keep_top_k(cls, graph, node, class_id, scores, bboxes):
        keep_top_k = node.attr('keep_top_k')
        keep_top_k = graph.make_node(
            'Constant',
            dtype=dtypes.ONNX.INT64,
            dims=[1, 1],
            value=[node.attr('keep_top_k')])
        # get min(topK, num_select)
        shape_select_num = graph.make_node('Shape', inputs=[scores])
        const_zero = graph.make_node(
            'Constant', dtype=dtypes.ONNX.INT64, value=[0])
        gather_select_num = graph.make_node(
            'Gather', inputs=[shape_select_num, const_zero], axis=0)
        unsqueeze_select_num = graph.make_node(
            'Unsqueeze', inputs=[gather_select_num], axes=[0])
        concat_topK_select_num = graph.make_node(
            'Concat', inputs=[unsqueeze_select_num, keep_top_k], axis=0)
        cast_concat_topK_select_num = graph.make_node(
            'Cast', inputs=[concat_topK_select_num], to=6)
        keep_top_k = graph.make_node(
            'ReduceMin', inputs=[cast_concat_topK_select_num], keepdims=0)
        # unsqueeze the indices to 1D tensor
        keep_top_k = graph.make_node('Unsqueeze', inputs=[keep_top_k], axes=[0])
        # cast the indices to INT64
        keep_top_k = graph.make_node('Cast', inputs=[keep_top_k], to=7)

        # select topk scores  indices
        keep_topk_scores, keep_topk_indices = graph.make_node(
            'TopK', inputs=[scores, keep_top_k], outputs=2)

        # gather topk label, scores, boxes
        gather_topk_scores = graph.make_node(
            'Gather', inputs=[scores, keep_topk_indices], axis=0)

        gather_topk_class = graph.make_node(
            'Gather', inputs=[class_id, keep_topk_indices], axis=1)

        # gather the boxes need to gather the boxes id, then get boxes
        gather_topk_boxes_id = graph.make_node(
            'Gather', [bbox_id, keep_topk_indices], axis=1)

        # squeeze the gather_topk_boxes_id to 1 dim
        squeeze_topk_boxes_id = graph.make_node(
            'Squeeze', inputs=[gather_topk_boxes_id], axes=[0, 2])

        gather_select_boxes = graph.make_node(
            'Gather', inputs=[bboxes, squeeze_topk_boxes_id], axis=1)

        # concat the final result
        # before concat need to cast the class to float
        cast_topk_class = graph.make_node(
            'Cast', inputs=[gather_topk_class], to=1)

        unsqueeze_topk_scores = graph.make_node(
            'Unsqueeze', inputs=[gather_topk_scores], axes=[0, 2])

        inputs_concat_final_results = [
            cast_topk_class, unsqueeze_topk_scores, gather_select_boxes
        ]

        sort_by_socre_results = graph.make_node(
            'Concat', inputs=inputs_concat_final_results, axis=2)

        # sort by class_id
        squeeze_cast_topk_class = graph.make_node(
            'Squeeze', inputs=[cast_topk_class], axes=[0, 2])

        neg_squeeze_cast_topk_class = graph.make_node(
            'Neg', inputs=[squeeze_cast_topk_class])

        data, indices = graph.make_node(
            'TopK', inputs=[neg_squeeze_cast_topk_class, keep_top_k], outputs=2)

        concat_final_results = graph.make_node(
            'Gather',
            inputs=[sort_by_socre_results, indices],
            outputs=[node.output('Out', 0)],
            axis=1)

    @classmethod
    def nms(cls, graph, node, scores, bboxes, times=0, **kw):
        result_name = node.output('Out', 0) + '_' + str(times)
        background = node.attr('background_label')
        normalized = node.attr('normalized')
        keep_top_k = node.attr('keep_top_k')
        nms_top_k = node.attr('nms_top_k')
        if nms_top_k < 0:
            nms_top_k = 10000
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

        node_nms_top_k = graph.make_node(
            'Constant',
            inputs=[],
            dtype=dtypes.ONNX.INT64,
            value=[np.int64(nms_top_k)])

        node_keep_top_k_2D = graph.make_node(
            'Constant',
            inputs=[],
            dtype=dtypes.ONNX.INT64,
            dims=[1, 1],
            value=[node.attr('keep_top_k')])

        # the paddle data format is x1,y1,x2,y2
        kwargs = {'center_point_box': 0}

        if not normalized:
            node_value_one = graph.make_node(
                'Constant',
                inputs=[],
                dims=[1],
                dtype=dtypes.ONNX.FLOAT,
                value=1.0)
            new_bboxes = graph.make_node(
                'Split', inputs=[bboxes], outputs=4, axis=2,
                split=[1, 1, 1, 1])
            new_xmax = graph.make_node(
                'Add', inputs=[new_bboxes[2], node_value_one])
            new_ymax = graph.make_node(
                'Add', inputs=[new_bboxes[3], node_value_one])
            new_bboxes = graph.make_node(
                'Concat',
                inputs=[new_bboxes[0], new_bboxes[1], new_xmax, new_ymax],
                axis=2)
            node_select_nms= graph.make_node(
               'NonMaxSuppression',
               inputs=[new_bboxes, scores, node_nms_top_k,\
                   node_iou_threshold, node_score_threshold])
        else:
            node_select_nms= graph.make_node(
                'NonMaxSuppression',
                inputs=[bboxes, scores, node_nms_top_k,\
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
            "Reshape", inputs=[scores, result_name + "@const_-1"])

        # get the shape of scores
        node_shape_scores = graph.make_node('Shape', inputs=scores)

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
            'Gather', inputs=[bboxes, node_squeeze_topk_boxes_id], axis=1)

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
            node_concat_final_indices = graph.make_node(
                'Squeeze',
                inputs=[node_gather_2_nonzero],
                outputs=node.output('Index'),
                axes=[0])
