## `paddle2onnx` Support Status

| Paddle Operator |  ONNX Opset Versions | support_status |
| --------------- | -------------------- | -------------- |
| abs | 1~12 |
| acos | 7~12 |
| arg_max | 1~12 |
| arg_min | 1~12 |
| arg_sort | 1~12 |
| asin | 7~12 |
| anchor_generator | 11~12 |
| assign_value | 1~12 |
| all | 6~12 |
| atan | 7~12 |
| any | 6~12 |
| batch_norm | 1~12 |
| bilinear_interp | 9~12 |
| bilinear_interp_v2 | 9~12 |
| bmm | 1~12 |
| box_coder | 7~12 |
| box_clip | 11~12 |
| cast | 1~12 |
| ceil | 1~13 | opset 6~13 limited supported |
| clip | 1~12 |
| cos | 7~12 |
| cosh | 9~12 |
| concat | 1~12 |
| conv2d | 1~12 |
| conv2d_transpose | 1~12 |
| conv3d | 1~12 |
| depthwise_conv2d_transpose | 1~12 |
| collect_fpn_proposals | 11~12 |
| cumsum | 11~12 |
| deformable_conv | 11~12 |
| depthwise_conv2d | 1~12 |
| distribute_fpn_proposals | 11~12 |
| dist | 7~12 |
| dropout | 7~12 |
| dot | 7~13 |
| elementwise_add | 7~12 |
| elementwise_div | 7~12 |
| elementwise_floordiv | 7~12 |
| elementwise_mul | 7~12 |
| elementwise_min | 7~12 |
| elementwise_max | 7~12 |
| elementwise_mod | 7~12 |
| elementwise_pow | 7~12 |
| elementwise_sub | 7~12 |
| equal | 1~12 |
| erf | 9~12 |
| exp | 1~12 |
| expand_as_v2 | 8~12 |
| expand_v2 | 8~12 |
| elu | 1~12 |
| fill_constant | 1~12 |
| fill_constant_batch_size_like  | 9~12 |
| fill_any_like | 9~12 |
| flatten2 | 1~12 |
| flatten_contiguous_range | 1~12 |
| floor | 1~12 |
| floor_mod | 7~12 |
| lod_reset | 1~12 |
| lstm | 9~12 |
| gather | 1~12 |  opset 1~10 limited supported |
| generate_proposals | 12~ |   |
| greater_equal | 12~ |   |
| group_norm | 1~12 |   |
| hardshrink | 9~12 |
| hardtanh | 6~12 |
| hard_sigmoid | 1~12 |
| hard_swish | 1~12 |
| has_nan | 9~12 |
| im2sequence | 1~12 |
| instance_norm | 1~12 |
| index_select | 1~12 |
| isinf | 10~12 |
| isnan | 9~12 |
| isfinite | 10~12 |
| layer_norm | 9~12 |
| leaky_relu | 1~12 |
| less_than | 1~12 | opset 7~12 limited supported
| less_equal| 12~ |
| log | 1~12 |
| log2 | 7~12 |
| lookup_table | 1~12 |
| lookup_table_v2 | 1~12 |
| logical_and | 1~12 |
| logical_not | 1~12 |
| logical_or | 1~12 | opset 7~12 limited supported |
| logsumexp | 1~12 |
| log10 | 7~12 |
| log1p | 7~12 |
| lookup_table | 1~12 |
| lookup_table_v2 | 1~12 |
| logical_and | 1~12 |
| logical_xor | 1~12 | opset 7~12 limited supported |
| logsigmoid | 1~12 |
| logsoftmax | 1~12 |
| matmul | 1~12 |
| matmul_v2 | 1~12 |
| mean | 1~12 |
| mul | 1~12 |
| muticlass_nms | 10~12 |
| muticlass_nms2 | 10~12 |
| mv | 1~12 |
| nearest_interp | 9~12 |
| nearest_interp_v2 | 9~12 |
| norm | 1~12 |
| numel | 1~12 |
| pad1d | 2~12 |
| pad2d | 1~12 |
| pad3d | 1~12 |
| pixel_shuffle | 11~12 |
| pool2d | 1~12 | limited supported |
| pool3d | 1~12 | limited supported |
| pow | 8~12 |
| prior_box | 1~12 |
| prelu | 1~12 |
| p_norm | 1~12 |
| range | 11~12 |
| reciprocal | 1~12 |
| reduce_mean | 1~12 |
| reduce_max | 1~12 |
| reduce_min | 1~12 |
| reduce_prod | 1~12 |
| reduce_sum | 1~12 |
| relu | 1~12 |
| relu6 | 1~12 |
| reshape2 | 5~12 |
| rnn | 1~12 |
| roi_align | 10~12 |
| round | 11~12 |
| rsqrt | 6~12 |
| softmax | 1~12 |
| scale | 1~12 | opset 1~6 limited supported |
| sequence_expand | 1~12 |
| selu | 6~12 |
| softmax_with_cross_entropy | 12 |
| softplus | 1~12 |
| softsign | 1~12 |
| shape | 1~12 |
| sigmoid | 1~12 |
| sign | 9~12 |
| sin | 7~12 |
| sinh | 9~12 |
| slice | 1~12 |
| split | 1~12 |
| squeeze2 | 1~12 |
| square | 7~12 |
| sqrt | 1~12 |
| stack | 1~12 |
| stride_slice | 1~12 |
| sum | 1~12 |
| swish | 1~12 |
| tanh | 1~12 |
| tile | 11~12 |
| top_k | 11~12 |
| top_k_v2 | 11~12 |
| transpose2 | 1~12 |
| uniform_random | 1~12 |
| uniform_random_batch_size_like | 1~12 |
| unsqueeze2 | 1~12 |
| where_index | 9~12 |
| yolo_box | 9~12 |
