// Auto-generated from ops.yaml + op_compat.yaml (version v3.4)
// DO NOT EDIT — regenerate via gen_op_info.py
#pragma once
#include <string>
#include <unordered_map>
#include <vector>

namespace paddle2onnx {
namespace pir {

// Op name mappings: fluid/legacy_name → phi/normalized_name
inline const std::unordered_map<std::string, std::string>&
GetOpNameMappings_v3.4() {
  static const std::unordered_map<std::string, std::string> m = {
    {"adadelta", "adadelta_"},
    {"adagrad", "adagrad_"},
    {"adam", "adam_"},
    {"adamax", "adamax_"},
    {"adamw", "adamw_"},
    {"elementwise_add", "add"},
    {"sum", "add_n"},
    {"reduce_all", "all"},
    {"reduce_amax", "amax"},
    {"reduce_amin", "amin"},
    {"reduce_any", "any"},
    {"range", "arange"},
    {"arg_max", "argmax"},
    {"arg_min", "argmin"},
    {"tensor_array_to_tensor", "array_to_tensor"},
    {"bicubic_interp_v2", "bicubic_interp"},
    {"bilinear_tensor_product", "bilinear"},
    {"bilinear_interp_v2", "bilinear_interp"},
    {"check_finite_and_unscale", "check_finite_and_unscale_"},
    {"crop_tensor", "crop"},
    {"softmax_with_cross_entropy", "cross_entropy_with_softmax"},
    {"determinant", "det"},
    {"diag_v2", "diag"},
    {"elementwise_div", "divide"},
    {"lookup_table_v2", "embedding"},
    {"expand_v2", "expand"},
    {"expand_as_v2", "expand_as"},
    {"exponential", "exponential_"},
    {"fetch_v2", "fetch"},
    {"fill_any", "fill"},
    {"flatten_contiguous_range", "flatten"},
    {"elementwise_floordiv", "floor_divide"},
    {"elementwise_fmax", "fmax"},
    {"elementwise_fmin", "fmin"},
    {"fill_constant", "full"},
    {"fill_constant_batch_size_like", "full_batch_size_like"},
    {"fill_any_like", "full_like"},
    {"fused_adam", "fused_adam_"},
    {"fused_bn_add_activation", "fused_bn_add_activation_"},
    {"gaussian_random", "gaussian"},
    {"generate_proposals_v2", "generate_proposals"},
    {"grid_sampler", "grid_sample"},
    {"hard_shrink", "hardshrink"},
    {"hard_sigmoid", "hardsigmoid"},
    {"hard_swish", "hardswish"},
    {"brelu", "hardtanh"},
    {"elementwise_heaviside", "heaviside"},
    {"hierarchical_sigmoid", "hsigmoid_loss"},
    {"isfinite_v2", "isfinite"},
    {"isinf_v2", "isinf"},
    {"isnan_v2", "isnan"},
    {"lamb", "lamb_"},
    {"lars_momentum", "lars_momentum_"},
    {"bilinear_interp", "legacy_bilinear_interp"},
    {"crop", "legacy_crop"},
    {"expand", "legacy_expand"},
    {"generate_proposals", "legacy_generate_proposals"},
    {"matmul", "legacy_matmul"},
    {"nearest_interp", "legacy_nearest_interp"},
    {"reshape", "legacy_reshape"},
    {"linear_interp_v2", "linear_interp"},
    {"matmul_v2", "matmul"},
    {"mul", "matmul_with_flatten"},
    {"reduce_max", "max"},
    {"elementwise_max", "maximum"},
    {"reduce_mean", "mean"},
    {"mean", "mean_all"},
    {"merged_momentum", "merged_momentum_"},
    {"reduce_min", "min"},
    {"elementwise_min", "minimum"},
    {"momentum", "momentum_"},
    {"elementwise_mul", "multiply"},
    {"nearest_interp_v2", "nearest_interp"},
    {"where_index", "nonzero"},
    {"size", "numel"},
    {"one_hot_v2", "one_hot"},
    {"reduce_prod", "prod"},
    {"graph_reindex", "reindex_graph"},
    {"elementwise_mod", "remainder"},
    {"reshape2", "reshape"},
    {"rmsprop", "rmsprop_"},
    {"graph_send_recv", "send_u_recv"},
    {"graph_send_ue_recv", "send_ue_recv"},
    {"graph_send_uv", "send_uv"},
    {"sgd", "sgd_"},
    {"share_data", "share_data_"},
    {"slogdeterminant", "slogdet"},
    {"squeeze2", "squeeze"},
    {"elementwise_sub", "subtract"},
    {"reduce_sum", "sum"},
    {"c_sync_calc_stream", "sync_calc_stream"},
    {"c_sync_comm_stream", "sync_comm_stream"},
    {"top_k_v2", "topk"},
    {"top_k", "topk_v1"},
    {"transpose2", "transpose"},
    {"trilinear_interp_v2", "trilinear_interp"},
    {"uniform_random", "uniform"},
    {"uniform_random_inplace", "uniform_inplace"},
    {"unsqueeze2", "unsqueeze"},
    {"update_loss_scaling", "update_loss_scaling_"},
    {"yolov3_loss", "yolo_loss"},
  };
  return m;
}

// Input name → positional index
inline const std::unordered_map<std::string, int>&
GetOpInputIndices_v3.4(const std::string& op_name) {
  static const std::unordered_map<std::string, std::unordered_map<std::string, int>> all = {
    {"abs", {
      {"x", 0},
    }},
    {"accuracy", {
      {"x", 0},
      {"indices", 1},
      {"label", 2},
    }},
    {"accuracy_check", {
      {"x", 0},
      {"y", 1},
    }},
    {"acos", {
      {"x", 0},
    }},
    {"acosh", {
      {"x", 0},
    }},
    {"adadelta_", {
      {"param", 0},
      {"grad", 1},
      {"avg_squared_grad", 2},
      {"avg_squared_update", 3},
      {"learning_rate", 4},
      {"master_param", 5},
    }},
    {"adagrad_", {
      {"param", 0},
      {"grad", 1},
      {"moment", 2},
      {"learning_rate", 3},
      {"master_param", 4},
    }},
    {"adam_", {
      {"param", 0},
      {"grad", 1},
      {"learning_rate", 2},
      {"moment1", 3},
      {"moment2", 4},
      {"moment2_max", 5},
      {"beta1_pow", 6},
      {"beta2_pow", 7},
      {"master_param", 8},
      {"skip_update", 9},
    }},
    {"adamax_", {
      {"param", 0},
      {"grad", 1},
      {"learning_rate", 2},
      {"moment", 3},
      {"inf_norm", 4},
      {"beta1_pow", 5},
      {"master_param", 6},
    }},
    {"adamw_", {
      {"param", 0},
      {"grad", 1},
      {"learning_rate", 2},
      {"moment1", 3},
      {"moment2", 4},
      {"moment2_max", 5},
      {"beta1_pow", 6},
      {"beta2_pow", 7},
      {"master_param", 8},
      {"skip_update", 9},
    }},
    {"add_position_encoding", {
      {"x", 0},
    }},
    {"addmm", {
      {"input", 0},
      {"x", 1},
      {"y", 2},
    }},
    {"affine_channel", {
      {"x", 0},
      {"scale", 1},
      {"bias", 2},
    }},
    {"affine_grid", {
      {"input", 0},
    }},
    {"all", {
      {"x", 0},
    }},
    {"all_gather", {
      {"x", 0},
    }},
    {"all_reduce", {
      {"x", 0},
    }},
    {"all_to_all", {
      {"x", 0},
    }},
    {"allclose", {
      {"x", 0},
      {"y", 1},
    }},
    {"amax", {
      {"x", 0},
    }},
    {"amin", {
      {"x", 0},
    }},
    {"aminmax", {
      {"x", 0},
    }},
    {"angle", {
      {"x", 0},
    }},
    {"any", {
      {"x", 0},
    }},
    {"ap_facade", {
      {"xs", 0},
    }},
    {"ap_trivial_fusion_begin", {
      {"xs", 0},
    }},
    {"ap_trivial_fusion_end", {
      {"xs", 0},
    }},
    {"ap_variadic", {
      {"xs", 0},
    }},
    {"apply_per_channel_scale", {
      {"x", 0},
      {"scales", 1},
    }},
    {"argmax", {
      {"x", 0},
    }},
    {"argmin", {
      {"x", 0},
    }},
    {"argsort", {
      {"x", 0},
    }},
    {"as_complex", {
      {"x", 0},
    }},
    {"as_real", {
      {"x", 0},
    }},
    {"as_strided", {
      {"input", 0},
    }},
    {"asgd_", {
      {"param", 0},
      {"grad", 1},
      {"learning_rate", 2},
      {"d", 3},
      {"y", 4},
      {"n", 5},
      {"master_param", 6},
    }},
    {"asin", {
      {"x", 0},
    }},
    {"asinh", {
      {"x", 0},
    }},
    {"assign_out_", {
      {"x", 0},
      {"output", 1},
    }},
    {"assign_pos", {
      {"x", 0},
      {"cum_count", 1},
      {"eff_num_len", 2},
    }},
    {"assign_value_", {
      {"output", 0},
    }},
    {"atan", {
      {"x", 0},
    }},
    {"atan2", {
      {"x", 0},
      {"y", 1},
    }},
    {"atanh", {
      {"x", 0},
    }},
    {"attention_lstm", {
      {"x", 0},
      {"c0", 1},
      {"h0", 2},
      {"attention_weight", 3},
      {"attention_bias", 4},
      {"attention_scalar", 5},
      {"attention_scalar_bias", 6},
      {"lstm_weight", 7},
      {"lstm_bias", 8},
    }},
    {"auc", {
      {"x", 0},
      {"label", 1},
      {"stat_pos", 2},
      {"stat_neg", 3},
      {"ins_tag_weight", 4},
    }},
    {"average_accumulates_", {
      {"param", 0},
      {"in_sum_1", 1},
      {"in_sum_2", 2},
      {"in_sum_3", 3},
      {"in_num_accumulates", 4},
      {"in_old_num_accumulates", 5},
      {"in_num_updates", 6},
    }},
    {"baddbmm", {
      {"input", 0},
      {"x", 1},
      {"y", 2},
    }},
    {"barrier", {
      {"x", 0},
    }},
    {"batch_fc", {
      {"input", 0},
      {"w", 1},
      {"bias", 2},
    }},
    {"batched_gemm", {
      {"lhs", 0},
      {"rhs", 1},
    }},
    {"bce_loss", {
      {"input", 0},
      {"label", 1},
    }},
    {"beam_search", {
      {"pre_ids", 0},
      {"pre_scores", 1},
      {"ids", 2},
      {"scores", 3},
    }},
    {"bernoulli", {
      {"x", 0},
    }},
    {"bicubic_interp", {
      {"x", 0},
      {"out_size", 1},
      {"size_tensor", 2},
      {"scale_tensor", 3},
    }},
    {"bilinear", {
      {"x", 0},
      {"y", 1},
      {"weight", 2},
      {"bias", 3},
    }},
    {"bilinear_interp", {
      {"x", 0},
      {"out_size", 1},
      {"size_tensor", 2},
      {"scale_tensor", 3},
    }},
    {"bincount", {
      {"x", 0},
      {"weights", 1},
    }},
    {"binomial", {
      {"count", 0},
      {"prob", 1},
    }},
    {"bipartite_match", {
      {"dist_mat", 0},
    }},
    {"bitwise_and", {
      {"x", 0},
      {"y", 1},
    }},
    {"bitwise_left_shift", {
      {"x", 0},
      {"y", 1},
    }},
    {"bitwise_not", {
      {"x", 0},
    }},
    {"bitwise_or", {
      {"x", 0},
      {"y", 1},
    }},
    {"bitwise_right_shift", {
      {"x", 0},
      {"y", 1},
    }},
    {"bitwise_xor", {
      {"x", 0},
      {"y", 1},
    }},
    {"bmm", {
      {"x", 0},
      {"y", 1},
    }},
    {"box_clip", {
      {"input", 0},
      {"im_info", 1},
    }},
    {"box_coder", {
      {"prior_box", 0},
      {"prior_box_var", 1},
      {"target_box", 2},
    }},
    {"broadcast", {
      {"x", 0},
    }},
    {"broadcast_tensors", {
      {"input", 0},
    }},
    {"build_src_rank_and_local_expert_id", {
      {"expert_num_global_tensor", 0},
    }},
    {"c_allreduce_sum", {
      {"x", 0},
    }},
    {"c_concat", {
      {"x", 0},
    }},
    {"c_identity", {
      {"x", 0},
    }},
    {"c_scatter", {
      {"x", 0},
    }},
    {"c_softmax_with_cross_entropy", {
      {"logits", 0},
      {"label", 1},
    }},
    {"c_split", {
      {"x", 0},
    }},
    {"cal_aux_loss", {
      {"gate_prob", 0},
      {"dispatch_mask", 1},
      {"tokens_mask", 2},
      {"dispatch_tokens_mask", 3},
    }},
    {"calc_reduced_attn_scores", {
      {"q", 0},
      {"k", 1},
      {"softmax_lse", 2},
    }},
    {"cast", {
      {"x", 0},
    }},
    {"ceil", {
      {"x", 0},
    }},
    {"celu", {
      {"x", 0},
    }},
    {"channel_shuffle", {
      {"x", 0},
    }},
    {"check_finite_and_unscale_", {
      {"x", 0},
      {"scale", 1},
    }},
    {"check_numerics", {
      {"tensor", 0},
    }},
    {"cholesky", {
      {"x", 0},
    }},
    {"cholesky_solve", {
      {"x", 0},
      {"y", 1},
    }},
    {"chunk_eval", {
      {"inference", 0},
      {"label", 1},
      {"seq_length", 2},
    }},
    {"class_center_sample", {
      {"label", 0},
    }},
    {"clip", {
      {"x", 0},
    }},
    {"clip_by_norm", {
      {"x", 0},
    }},
    {"coalesce_tensor", {
      {"input", 0},
    }},
    {"collect_fpn_proposals", {
      {"multi_level_rois", 0},
      {"multi_level_scores", 1},
      {"multi_level_rois_num", 2},
    }},
    {"complex", {
      {"real", 0},
      {"imag", 1},
    }},
    {"concat", {
      {"x", 0},
    }},
    {"conj", {
      {"x", 0},
    }},
    {"conv2d", {
      {"input", 0},
      {"filter", 1},
    }},
    {"conv2d_transpose", {
      {"x", 0},
      {"filter", 1},
    }},
    {"conv2d_transpose_bias", {
      {"x", 0},
      {"filter", 1},
      {"bias", 2},
    }},
    {"conv3d", {
      {"input", 0},
      {"filter", 1},
    }},
    {"conv3d_transpose", {
      {"x", 0},
      {"filter", 1},
    }},
    {"copy_to", {
      {"x", 0},
    }},
    {"copysign", {
      {"x", 0},
      {"y", 1},
    }},
    {"correlation", {
      {"input1", 0},
      {"input2", 1},
    }},
    {"cos", {
      {"x", 0},
    }},
    {"cosh", {
      {"x", 0},
    }},
    {"crf_decoding", {
      {"emission", 0},
      {"transition", 1},
      {"label", 2},
      {"length", 3},
    }},
    {"crop", {
      {"x", 0},
    }},
    {"cross", {
      {"x", 0},
      {"y", 1},
    }},
    {"cross_entropy_with_softmax", {
      {"input", 0},
      {"label", 1},
    }},
    {"cross_entropy_with_softmax_bwd_w_downcast", {
      {"label", 0},
      {"softmax", 1},
      {"loss_grad", 2},
    }},
    {"ctc_align", {
      {"input", 0},
      {"input_length", 1},
    }},
    {"cudnn_lstm", {
      {"x", 0},
      {"init_h", 1},
      {"init_c", 2},
      {"w", 3},
      {"weight_list", 4},
      {"sequence_length", 5},
    }},
    {"cummax", {
      {"x", 0},
    }},
    {"cummin", {
      {"x", 0},
    }},
    {"cumprod", {
      {"x", 0},
    }},
    {"cumsum", {
      {"x", 0},
    }},
    {"cvm", {
      {"x", 0},
      {"cvm", 1},
    }},
    {"data", {
      {"name", 0},
      {"shape", 1},
      {"dtype", 2},
      {"place", 3},
    }},
    {"decayed_adagrad", {
      {"param", 0},
      {"grad", 1},
      {"moment", 2},
      {"learning_rate", 3},
    }},
    {"decode_jpeg", {
      {"x", 0},
    }},
    {"deformable_conv", {
      {"x", 0},
      {"offset", 1},
      {"filter", 2},
      {"mask", 3},
    }},
    {"depend", {
      {"x", 0},
      {"dep", 1},
    }},
    {"depthwise_conv2d", {
      {"input", 0},
      {"filter", 1},
    }},
    {"depthwise_conv2d_bias", {
      {"input", 0},
      {"filter", 1},
      {"bias", 2},
    }},
    {"depthwise_conv2d_transpose", {
      {"x", 0},
      {"filter", 1},
    }},
    {"depthwise_conv3d_bias", {
      {"input", 0},
      {"filter", 1},
      {"bias", 2},
    }},
    {"dequantize_abs_max", {
      {"x", 0},
      {"scale", 1},
    }},
    {"dequantize_log", {
      {"x", 0},
      {"dict", 1},
    }},
    {"det", {
      {"x", 0},
    }},
    {"dgc", {
      {"u", 0},
      {"v", 1},
      {"grad", 2},
      {"param", 3},
      {"current_step", 4},
      {"nranks", 5},
    }},
    {"dgc_clip_by_norm", {
      {"x", 0},
      {"current_step", 1},
    }},
    {"dgc_momentum", {
      {"param", 0},
      {"grad", 1},
      {"velocity", 2},
      {"learning_rate", 3},
      {"master_param", 4},
      {"current_step_tensor", 5},
      {"nranks_tensor", 6},
    }},
    {"diag", {
      {"x", 0},
    }},
    {"diag_embed", {
      {"input", 0},
    }},
    {"diagonal", {
      {"x", 0},
    }},
    {"digamma", {
      {"x", 0},
    }},
    {"dirichlet", {
      {"alpha", 0},
    }},
    {"disable_check_model_nan_inf", {
      {"x", 0},
    }},
    {"dist", {
      {"x", 0},
      {"y", 1},
    }},
    {"dot", {
      {"x", 0},
      {"y", 1},
    }},
    {"dpsgd", {
      {"param", 0},
      {"grad", 1},
      {"learning_rate", 2},
    }},
    {"dropout", {
      {"x", 0},
      {"seed_tensor", 1},
    }},
    {"edit_distance", {
      {"hyps", 0},
      {"refs", 1},
      {"hypslength", 2},
      {"refslength", 3},
    }},
    {"eig", {
      {"x", 0},
    }},
    {"eigh", {
      {"x", 0},
    }},
    {"eigvals", {
      {"x", 0},
    }},
    {"eigvalsh", {
      {"x", 0},
    }},
    {"elu", {
      {"x", 0},
    }},
    {"embedding_grad_add_to", {
      {"token_indices", 0},
      {"main_grad_", 1},
      {"out_grad", 2},
    }},
    {"embedding_with_scaled_gradient", {
      {"x", 0},
      {"weight", 1},
    }},
    {"empty", {
      {"shape", 0},
      {"dtype", 1},
      {"place", 2},
    }},
    {"empty_like", {
      {"x", 0},
    }},
    {"enable_check_model_nan_inf", {
      {"x", 0},
    }},
    {"equal_all", {
      {"x", 0},
      {"y", 1},
    }},
    {"erf", {
      {"x", 0},
    }},
    {"erfinv", {
      {"x", 0},
    }},
    {"exp", {
      {"x", 0},
    }},
    {"expand", {
      {"x", 0},
    }},
    {"expand_as", {
      {"x", 0},
      {"y", 1},
    }},
    {"expand_modality_expert_id", {
      {"expert_id", 0},
    }},
    {"expm1", {
      {"x", 0},
    }},
    {"exponential_", {
      {"x", 0},
    }},
    {"eye", {
      {"num_rows", 0},
      {"num_columns", 1},
      {"dtype", 2},
      {"place", 3},
    }},
    {"fake_channel_wise_dequantize_max_abs", {
      {"x", 0},
      {"scales", 1},
    }},
    {"fake_channel_wise_quantize_abs_max", {
      {"x", 0},
    }},
    {"fake_channel_wise_quantize_dequantize_abs_max", {
      {"x", 0},
    }},
    {"fake_dequantize_max_abs", {
      {"x", 0},
      {"scale", 1},
    }},
    {"fake_quantize_abs_max", {
      {"x", 0},
    }},
    {"fake_quantize_dequantize_abs_max", {
      {"x", 0},
    }},
    {"fake_quantize_dequantize_moving_average_abs_max", {
      {"x", 0},
      {"in_scale", 1},
      {"in_accum", 2},
      {"in_state", 3},
    }},
    {"fake_quantize_moving_average_abs_max", {
      {"x", 0},
      {"in_scale", 1},
      {"in_accum", 2},
      {"in_state", 3},
    }},
    {"fake_quantize_range_abs_max", {
      {"x", 0},
      {"in_scale", 1},
      {"iter", 2},
    }},
    {"fast_ln", {
      {"x", 0},
      {"scale", 1},
      {"bias", 2},
    }},
    {"fast_rms_norm", {
      {"x", 0},
      {"scale", 1},
    }},
    {"fft_c2c", {
      {"x", 0},
    }},
    {"fft_c2r", {
      {"x", 0},
    }},
    {"fft_r2c", {
      {"x", 0},
    }},
    {"fill", {
      {"x", 0},
    }},
    {"fill_diagonal", {
      {"x", 0},
    }},
    {"fill_diagonal_tensor", {
      {"x", 0},
      {"y", 1},
    }},
    {"flash_attn", {
      {"q", 0},
      {"k", 1},
      {"v", 2},
      {"fixed_seed_offset", 3},
      {"attn_mask", 4},
    }},
    {"flash_attn_qkvpacked", {
      {"qkv", 0},
      {"fixed_seed_offset", 1},
      {"attn_mask", 2},
    }},
    {"flash_attn_unpadded", {
      {"q", 0},
      {"k", 1},
      {"v", 2},
      {"cu_seqlens_q", 3},
      {"cu_seqlens_k", 4},
      {"fixed_seed_offset", 5},
      {"attn_mask", 6},
    }},
    {"flash_attn_v3", {
      {"q", 0},
      {"k", 1},
      {"v", 2},
      {"q_v_", 3},
      {"q_descale_", 4},
      {"k_descale_", 5},
      {"v_descale_", 6},
    }},
    {"flash_attn_v3_varlen", {
      {"q", 0},
      {"k", 1},
      {"v", 2},
      {"cu_seqlens_q", 3},
      {"cu_seqlens_k", 4},
      {"seqused_q", 5},
      {"seqused_k", 6},
      {"qv", 7},
      {"q_descale", 8},
      {"k_descale", 9},
      {"v_descale", 10},
    }},
    {"flash_attn_varlen_qkvpacked", {
      {"qkv", 0},
      {"cu_seqlens_q", 1},
      {"cu_seqlens_k", 2},
      {"fixed_seed_offset", 3},
      {"attn_mask", 4},
    }},
    {"flashmask_attention", {
      {"q", 0},
      {"k", 1},
      {"v", 2},
      {"startend_row_indices", 3},
      {"fixed_seed_offset", 4},
    }},
    {"flashmask_attention_v2", {
      {"q", 0},
      {"k", 1},
      {"v", 2},
      {"startend_row_indices", 3},
      {"block_mask", 4},
      {"unique_id", 5},
    }},
    {"flashmask_get_unique_id", {
      {"x", 0},
    }},
    {"flatten", {
      {"x", 0},
    }},
    {"flip", {
      {"x", 0},
    }},
    {"floor", {
      {"x", 0},
    }},
    {"fmax", {
      {"x", 0},
      {"y", 1},
    }},
    {"fmin", {
      {"x", 0},
      {"y", 1},
    }},
    {"fold", {
      {"x", 0},
    }},
    {"fp8_gemm_blockwise_", {
      {"A", 0},
      {"A_scale", 1},
      {"B", 2},
      {"B_scale", 3},
      {"input_result", 4},
      {"bias", 5},
      {"pre_gelu", 6},
      {"workspace", 7},
    }},
    {"fp8_quant_blockwise", {
      {"x", 0},
    }},
    {"fractional_max_pool2d", {
      {"x", 0},
    }},
    {"fractional_max_pool3d", {
      {"x", 0},
    }},
    {"frame", {
      {"x", 0},
    }},
    {"frobenius_norm", {
      {"x", 0},
    }},
    {"ftrl", {
      {"param", 0},
      {"squared_accumulator", 1},
      {"linear_accumulator", 2},
      {"grad", 3},
      {"learning_rate", 4},
    }},
    {"full", {
      {"shape", 0},
      {"value", 1},
      {"dtype", 2},
      {"place", 3},
    }},
    {"full_", {
      {"output", 0},
    }},
    {"full_batch_size_like", {
      {"input", 0},
    }},
    {"full_int_array", {
      {"value", 0},
      {"dtype", 1},
      {"place", 2},
    }},
    {"full_like", {
      {"x", 0},
    }},
    {"full_with_tensor", {
      {"value", 0},
    }},
    {"fused_batch_norm_act", {
      {"x", 0},
      {"scale", 1},
      {"bias", 2},
      {"mean", 3},
      {"variance", 4},
    }},
    {"fused_bn_add_activation", {
      {"x", 0},
      {"z", 1},
      {"scale", 2},
      {"bias", 3},
      {"mean", 4},
      {"variance", 5},
    }},
    {"fused_rms_norm_ext", {
      {"x", 0},
      {"scale", 1},
    }},
    {"fused_rms_norm_quant", {
      {"x", 0},
      {"bias", 1},
      {"residual", 2},
      {"norm_weight", 3},
      {"norm_bias", 4},
    }},
    {"fused_softmax_mask", {
      {"x", 0},
      {"mask", 1},
    }},
    {"fused_softmax_mask_upper_triangle", {
      {"X", 0},
    }},
    {"gammaincc", {
      {"x", 0},
      {"y", 1},
    }},
    {"gammaln", {
      {"x", 0},
    }},
    {"gather", {
      {"x", 0},
      {"index", 1},
    }},
    {"gather_nd", {
      {"x", 0},
      {"index", 1},
    }},
    {"gather_tree", {
      {"ids", 0},
      {"parents", 1},
    }},
    {"gaussian", {
      {"shape", 0},
      {"mean", 1},
      {"std", 2},
      {"seed", 3},
      {"dtype", 4},
      {"place", 5},
    }},
    {"gaussian_inplace", {
      {"x", 0},
    }},
    {"gelu", {
      {"x", 0},
    }},
    {"generate_proposals", {
      {"scores", 0},
      {"bbox_deltas", 1},
      {"im_shape", 2},
      {"anchors", 3},
      {"variances", 4},
    }},
    {"global_gather", {
      {"x", 0},
      {"local_count", 1},
      {"global_count", 2},
    }},
    {"global_scatter", {
      {"x", 0},
      {"local_count", 1},
      {"global_count", 2},
    }},
    {"graph_khop_sampler", {
      {"row", 0},
      {"colptr", 1},
      {"x", 2},
      {"eids", 3},
    }},
    {"graph_sample_neighbors", {
      {"row", 0},
      {"colptr", 1},
      {"x", 2},
      {"eids", 3},
      {"perm_buffer", 4},
    }},
    {"grid_sample", {
      {"x", 0},
      {"grid", 1},
    }},
    {"group_norm", {
      {"x", 0},
      {"scale", 1},
      {"bias", 2},
    }},
    {"gru", {
      {"input", 0},
      {"h0", 1},
      {"weight", 2},
      {"bias", 3},
    }},
    {"gru_unit", {
      {"input", 0},
      {"hidden_prev", 1},
      {"weight", 2},
      {"bias", 3},
    }},
    {"gumbel_softmax", {
      {"x", 0},
    }},
    {"hardshrink", {
      {"x", 0},
    }},
    {"hardsigmoid", {
      {"x", 0},
    }},
    {"hardtanh", {
      {"x", 0},
    }},
    {"heaviside", {
      {"x", 0},
      {"y", 1},
    }},
    {"hinge_loss", {
      {"logits", 0},
      {"labels", 1},
    }},
    {"histogram", {
      {"input", 0},
      {"weight", 1},
    }},
    {"hsigmoid_loss", {
      {"x", 0},
      {"label", 1},
      {"w", 2},
      {"bias", 3},
      {"path", 4},
      {"code", 5},
    }},
    {"huber_loss", {
      {"input", 0},
      {"label", 1},
    }},
    {"i0", {
      {"x", 0},
    }},
    {"i0e", {
      {"x", 0},
    }},
    {"i1", {
      {"x", 0},
    }},
    {"i1e", {
      {"x", 0},
    }},
    {"identity_loss", {
      {"x", 0},
    }},
    {"im2sequence", {
      {"x", 0},
      {"y", 1},
    }},
    {"imag", {
      {"x", 0},
    }},
    {"increment", {
      {"x", 0},
    }},
    {"index_add", {
      {"x", 0},
      {"index", 1},
      {"add_value", 2},
    }},
    {"index_elementwise_get", {
      {"x", 0},
      {"index", 1},
    }},
    {"index_elementwise_put", {
      {"x", 0},
      {"index", 1},
    }},
    {"index_elementwise_put_with_tensor", {
      {"x", 0},
      {"index", 1},
      {"value", 2},
    }},
    {"index_fill", {
      {"x", 0},
      {"index", 1},
    }},
    {"index_put", {
      {"x", 0},
      {"indices", 1},
      {"value", 2},
    }},
    {"index_sample", {
      {"x", 0},
      {"index", 1},
    }},
    {"index_select", {
      {"x", 0},
      {"index", 1},
    }},
    {"index_select_strided", {
      {"x", 0},
    }},
    {"instance_norm", {
      {"x", 0},
      {"scale", 1},
      {"bias", 2},
    }},
    {"int_bincount", {
      {"x", 0},
    }},
    {"interp_antialias", {
      {"x", 0},
      {"out_size", 1},
      {"size_tensor", 2},
      {"scale_tensor", 3},
    }},
    {"inverse", {
      {"x", 0},
    }},
    {"is_empty", {
      {"x", 0},
    }},
    {"isclose", {
      {"x", 0},
      {"y", 1},
    }},
    {"isfinite", {
      {"x", 0},
    }},
    {"isinf", {
      {"x", 0},
    }},
    {"isnan", {
      {"x", 0},
    }},
    {"kldiv_loss", {
      {"x", 0},
      {"label", 1},
    }},
    {"kron", {
      {"x", 0},
      {"y", 1},
    }},
    {"kthvalue", {
      {"x", 0},
    }},
    {"l1_norm", {
      {"x", 0},
    }},
    {"label_smooth", {
      {"label", 0},
      {"prior_dist", 1},
    }},
    {"lamb_", {
      {"param", 0},
      {"grad", 1},
      {"learning_rate", 2},
      {"moment1", 3},
      {"moment2", 4},
      {"beta1_pow", 5},
      {"beta2_pow", 6},
      {"master_param", 7},
      {"skip_update", 8},
    }},
    {"layer_norm", {
      {"x", 0},
      {"scale", 1},
      {"bias", 2},
    }},
    {"leaky_relu", {
      {"x", 0},
    }},
    {"lerp", {
      {"x", 0},
      {"y", 1},
      {"weight", 2},
    }},
    {"lgamma", {
      {"x", 0},
    }},
    {"limit_by_capacity", {
      {"expert_count", 0},
      {"capacity", 1},
    }},
    {"linear_interp", {
      {"x", 0},
      {"out_size", 1},
      {"size_tensor", 2},
      {"scale_tensor", 3},
    }},
    {"linear_v2", {
      {"input", 0},
      {"weight", 1},
      {"bias", 2},
    }},
    {"linspace", {
      {"start", 0},
      {"stop", 1},
      {"number", 2},
    }},
    {"llm_int8_linear", {
      {"x", 0},
      {"weight", 1},
      {"bias", 2},
      {"weight_scale", 3},
    }},
    {"log", {
      {"x", 0},
    }},
    {"log10", {
      {"x", 0},
    }},
    {"log1p", {
      {"x", 0},
    }},
    {"log2", {
      {"x", 0},
    }},
    {"log_loss", {
      {"input", 0},
      {"label", 1},
    }},
    {"log_softmax", {
      {"x", 0},
    }},
    {"logcumsumexp", {
      {"x", 0},
    }},
    {"logical_and", {
      {"x", 0},
      {"y", 1},
    }},
    {"logical_not", {
      {"x", 0},
    }},
    {"logical_or", {
      {"x", 0},
      {"y", 1},
    }},
    {"logical_xor", {
      {"x", 0},
      {"y", 1},
    }},
    {"logit", {
      {"x", 0},
    }},
    {"logsigmoid", {
      {"x", 0},
    }},
    {"logspace", {
      {"start", 0},
      {"stop", 1},
      {"num", 2},
      {"base", 3},
    }},
    {"logsumexp", {
      {"x", 0},
    }},
    {"lookup_table_dequant", {
      {"w", 0},
      {"ids", 1},
    }},
    {"lp_pool2d", {
      {"x", 0},
    }},
    {"lstm", {
      {"input", 0},
      {"h0", 1},
      {"c0", 2},
      {"weight", 3},
      {"bias", 4},
    }},
    {"lstsq", {
      {"x", 0},
      {"y", 1},
    }},
    {"lu", {
      {"x", 0},
    }},
    {"lu_solve", {
      {"b", 0},
      {"lu", 1},
      {"pivots", 2},
    }},
    {"lu_unpack", {
      {"x", 0},
      {"y", 1},
    }},
    {"margin_cross_entropy", {
      {"logits", 0},
      {"label", 1},
    }},
    {"masked_fill", {
      {"x", 0},
      {"mask", 1},
      {"value", 2},
    }},
    {"masked_multihead_attention_", {
      {"x", 0},
      {"cache_kv", 1},
      {"bias", 2},
      {"src_mask", 3},
      {"cum_offsets", 4},
      {"sequence_lengths", 5},
      {"rotary_tensor", 6},
      {"beam_cache_offset", 7},
      {"qkv_out_scale", 8},
      {"out_shift", 9},
      {"out_smooth", 10},
    }},
    {"masked_scatter", {
      {"x", 0},
      {"mask", 1},
      {"value", 2},
    }},
    {"masked_select", {
      {"x", 0},
      {"mask", 1},
    }},
    {"match_matrix_tensor", {
      {"x", 0},
      {"y", 1},
      {"w", 2},
    }},
    {"matrix_nms", {
      {"bboxes", 0},
      {"scores", 1},
    }},
    {"matrix_power", {
      {"x", 0},
    }},
    {"matrix_rank", {
      {"x", 0},
    }},
    {"matrix_rank_atol_rtol", {
      {"x", 0},
      {"atol", 1},
      {"rtol", 2},
    }},
    {"matrix_rank_tol", {
      {"x", 0},
      {"atol_tensor", 1},
    }},
    {"max", {
      {"x", 0},
    }},
    {"max_pool2d_with_index", {
      {"x", 0},
    }},
    {"max_pool3d_with_index", {
      {"x", 0},
    }},
    {"max_with_index", {
      {"x", 0},
    }},
    {"maxout", {
      {"x", 0},
    }},
    {"mean", {
      {"x", 0},
    }},
    {"mean_all", {
      {"x", 0},
    }},
    {"median", {
      {"x", 0},
    }},
    {"memcpy_d2h", {
      {"x", 0},
    }},
    {"memcpy_h2d", {
      {"x", 0},
    }},
    {"memory_efficient_attention", {
      {"query", 0},
      {"key", 1},
      {"value", 2},
      {"bias", 3},
      {"cu_seqlens_q", 4},
      {"cu_seqlens_k", 5},
      {"causal_diagonal", 6},
      {"seqlen_k", 7},
    }},
    {"merge_selected_rows", {
      {"x", 0},
    }},
    {"merged_adam_", {
      {"param", 0},
      {"grad", 1},
      {"learning_rate", 2},
      {"moment1", 3},
      {"moment2", 4},
      {"moment2_max", 5},
      {"beta1_pow", 6},
      {"beta2_pow", 7},
      {"master_param", 8},
    }},
    {"merged_momentum_", {
      {"param", 0},
      {"grad", 1},
      {"velocity", 2},
      {"learning_rate", 3},
      {"master_param", 4},
    }},
    {"meshgrid", {
      {"inputs", 0},
    }},
    {"min_with_index", {
      {"x", 0},
    }},
    {"mish", {
      {"x", 0},
    }},
    {"mode", {
      {"x", 0},
    }},
    {"moe_combine", {
      {"x", 0},
      {"combine_weights", 1},
      {"scatter_index", 2},
    }},
    {"moe_combine_auto", {
      {"x", 0},
      {"combine_weights", 1},
      {"scatter_index", 2},
    }},
    {"moe_combine_no_weight", {
      {"x", 0},
      {"combine_weight", 1},
      {"scatter_index", 2},
    }},
    {"moe_gate_dispatch", {
      {"x", 0},
      {"gate_logits", 1},
      {"corr_bias", 2},
    }},
    {"moe_gate_dispatch_and_quant", {
      {"x", 0},
      {"gate_logits", 1},
      {"corr_bias", 2},
    }},
    {"moe_gate_dispatch_auto", {
      {"x", 0},
      {"gate_logits", 1},
      {"corr_bias", 2},
    }},
    {"moe_gate_dispatch_partial_nosoftmaxtopk", {
      {"x", 0},
      {"combine_weights", 1},
      {"expert_id", 2},
    }},
    {"moe_gate_dispatch_permute", {
      {"x", 0},
      {"gate_logits", 1},
      {"corr_bias", 2},
    }},
    {"moe_permute", {
      {"hidden_states", 0},
      {"scale", 1},
      {"expert_routemap_topk", 2},
      {"expert_prob_topk", 3},
    }},
    {"moe_unpermute", {
      {"hidden_states_unzipped", 0},
      {"zipped_expertwise_rowmap", 1},
      {"expert_routemap_topk", 2},
      {"token_prob_unzipped", 3},
    }},
    {"momentum_", {
      {"param", 0},
      {"grad", 1},
      {"velocity", 2},
      {"learning_rate", 3},
      {"master_param", 4},
    }},
    {"mp_allreduce_sum", {
      {"x", 0},
    }},
    {"multi_dot", {
      {"x", 0},
    }},
    {"multiclass_nms3", {
      {"bboxes", 0},
      {"scores", 1},
      {"rois_num", 2},
    }},
    {"multinomial", {
      {"x", 0},
    }},
    {"multiplex", {
      {"inputs", 0},
      {"index", 1},
    }},
    {"mv", {
      {"x", 0},
      {"vec", 1},
    }},
    {"nadam_", {
      {"param", 0},
      {"grad", 1},
      {"learning_rate", 2},
      {"momentum_decay_pow", 3},
      {"beta2_pow", 4},
      {"mu_product", 5},
      {"moment1", 6},
      {"moment2", 7},
      {"master_param", 8},
    }},
    {"nanmedian", {
      {"x", 0},
    }},
    {"nansum", {
      {"x", 0},
    }},
    {"nearest_interp", {
      {"x", 0},
      {"out_size", 1},
      {"size_tensor", 2},
      {"scale_tensor", 3},
    }},
    {"nextafter", {
      {"x", 0},
      {"y", 1},
    }},
    {"nll_loss", {
      {"input", 0},
      {"label", 1},
      {"weight", 2},
    }},
    {"nms", {
      {"x", 0},
    }},
    {"nonzero", {
      {"condition", 0},
    }},
    {"norm", {
      {"x", 0},
    }},
    {"npu_identity", {
      {"x", 0},
    }},
    {"number_count", {
      {"numbers", 0},
    }},
    {"numel", {
      {"x", 0},
    }},
    {"one_hot", {
      {"x", 0},
    }},
    {"ones", {
      {"shape", 0},
      {"dtype", 1},
      {"place", 2},
    }},
    {"ones_like", {
      {"x", 0},
    }},
    {"overlap_add", {
      {"x", 0},
    }},
    {"p_norm", {
      {"x", 0},
    }},
    {"pad", {
      {"x", 0},
    }},
    {"pad3d", {
      {"x", 0},
    }},
    {"partial_allgather", {
      {"x", 0},
    }},
    {"partial_concat", {
      {"x", 0},
    }},
    {"partial_sum", {
      {"x", 0},
    }},
    {"pixel_shuffle", {
      {"x", 0},
    }},
    {"pixel_unshuffle", {
      {"x", 0},
    }},
    {"poisson", {
      {"x", 0},
    }},
    {"polygamma", {
      {"x", 0},
    }},
    {"pool2d", {
      {"x", 0},
    }},
    {"pool3d", {
      {"x", 0},
    }},
    {"pow", {
      {"x", 0},
    }},
    {"prelu", {
      {"x", 0},
      {"alpha", 1},
    }},
    {"prior_box", {
      {"input", 0},
      {"image", 1},
    }},
    {"prod", {
      {"x", 0},
    }},
    {"prune_gate_by_capacity", {
      {"gate_idx", 0},
      {"expert_count", 1},
    }},
    {"psroi_pool", {
      {"x", 0},
      {"boxes", 1},
      {"boxes_num", 2},
    }},
    {"put_along_axis", {
      {"arr", 0},
      {"indices", 1},
      {"values", 2},
    }},
    {"pyramid_hash", {
      {"x", 0},
      {"w", 1},
      {"white_list", 2},
      {"black_list", 3},
    }},
    {"qr", {
      {"x", 0},
    }},
    {"radam_", {
      {"param", 0},
      {"grad", 1},
      {"learning_rate", 2},
      {"beta1_pow", 3},
      {"beta2_pow", 4},
      {"rho", 5},
      {"moment1", 6},
      {"moment2", 7},
      {"master_param", 8},
    }},
    {"randint", {
      {"low", 0},
      {"high", 1},
      {"shape", 2},
      {"dtype", 3},
      {"place", 4},
    }},
    {"random", {
      {"x", 0},
    }},
    {"random_routing", {
      {"prob", 0},
      {"topk_value", 1},
      {"topk_idx", 2},
    }},
    {"randperm", {
      {"n", 0},
      {"dtype", 1},
      {"place", 2},
    }},
    {"rank_attention", {
      {"x", 0},
      {"rank_offset", 1},
      {"rank_param", 2},
    }},
    {"read_file", {
      {"""", 0},
      {"dtype", 1},
      {"place", 2},
    }},
    {"real", {
      {"x", 0},
    }},
    {"reciprocal", {
      {"x", 0},
    }},
    {"reduce", {
      {"x", 0},
    }},
    {"reduce_as", {
      {"x", 0},
      {"target", 1},
    }},
    {"reduce_scatter", {
      {"x", 0},
    }},
    {"reindex_graph", {
      {"x", 0},
      {"neighbors", 1},
      {"count", 2},
      {"hashtable_value", 3},
      {"hashtable_index", 4},
    }},
    {"relu", {
      {"x", 0},
    }},
    {"relu6", {
      {"x", 0},
    }},
    {"renorm", {
      {"x", 0},
    }},
    {"repeat_interleave", {
      {"x", 0},
    }},
    {"repeat_interleave_with_tensor_index", {
      {"x", 0},
      {"repeats", 1},
    }},
    {"reshape", {
      {"x", 0},
    }},
    {"restrict_nonzero", {
      {"condition", 0},
    }},
    {"reverse", {
      {"x", 0},
    }},
    {"rint", {
      {"x", 0},
    }},
    {"rms_norm", {
      {"x", 0},
      {"scale", 1},
    }},
    {"rmsprop_", {
      {"param", 0},
      {"mean_square", 1},
      {"grad", 2},
      {"moment", 3},
      {"learning_rate", 4},
      {"mean_grad", 5},
      {"master_param", 6},
    }},
    {"rnn", {
      {"x", 0},
      {"pre_state", 1},
      {"weight_list", 2},
      {"sequence_length", 3},
      {"dropout_state_in", 4},
    }},
    {"roi_align", {
      {"x", 0},
      {"boxes", 1},
      {"boxes_num", 2},
    }},
    {"roi_pool", {
      {"x", 0},
      {"boxes", 1},
      {"boxes_num", 2},
    }},
    {"roll", {
      {"x", 0},
    }},
    {"round", {
      {"x", 0},
    }},
    {"rprop_", {
      {"param", 0},
      {"grad", 1},
      {"prev", 2},
      {"learning_rate", 3},
      {"master_param", 4},
      {"learning_rate_range", 5},
      {"etas", 6},
    }},
    {"rrelu", {
      {"x", 0},
    }},
    {"rsqrt", {
      {"x", 0},
    }},
    {"scale", {
      {"x", 0},
    }},
    {"scatter", {
      {"x", 0},
      {"index", 1},
      {"updates", 2},
    }},
    {"scatter_nd_add", {
      {"x", 0},
      {"index", 1},
      {"updates", 2},
    }},
    {"searchsorted", {
      {"sorted_sequence", 0},
      {"values", 1},
    }},
    {"segment_pool", {
      {"x", 0},
      {"segment_ids", 1},
    }},
    {"selu", {
      {"x", 0},
    }},
    {"send_u_recv", {
      {"x", 0},
      {"src_index", 1},
      {"dst_index", 2},
    }},
    {"send_ue_recv", {
      {"x", 0},
      {"y", 1},
      {"src_index", 2},
      {"dst_index", 3},
    }},
    {"send_uv", {
      {"x", 0},
      {"y", 1},
      {"src_index", 2},
      {"dst_index", 3},
    }},
    {"sequence_conv", {
      {"x", 0},
      {"padding_data", 1},
      {"filter", 2},
    }},
    {"sequence_mask", {
      {"x", 0},
    }},
    {"sequence_pool", {
      {"x", 0},
    }},
    {"set", {
      {"x", 0},
      {"source", 1},
    }},
    {"set_value_with_tensor", {
      {"x", 0},
      {"values", 1},
    }},
    {"sgd_", {
      {"param", 0},
      {"learning_rate", 1},
      {"grad", 2},
      {"master_param", 3},
    }},
    {"shape", {
      {"input", 0},
    }},
    {"shape64", {
      {"input", 0},
    }},
    {"shard_index", {
      {"input", 0},
    }},
    {"share_data", {
      {"x", 0},
    }},
    {"shuffle_batch", {
      {"x", 0},
      {"seed", 1},
    }},
    {"shuffle_channel", {
      {"x", 0},
    }},
    {"sigmoid", {
      {"x", 0},
    }},
    {"sigmoid_cross_entropy_with_logits", {
      {"x", 0},
      {"label", 1},
      {"pos_weight", 2},
    }},
    {"sign", {
      {"x", 0},
    }},
    {"silu", {
      {"x", 0},
    }},
    {"sin", {
      {"x", 0},
    }},
    {"sinh", {
      {"x", 0},
    }},
    {"slice", {
      {"input", 0},
    }},
    {"slogdet", {
      {"x", 0},
    }},
    {"slogdet_v2", {
      {"x", 0},
    }},
    {"slow_conv2d_dilated", {
      {"input", 0},
      {"filter", 1},
      {"bias", 2},
    }},
    {"slow_conv3d_dilated", {
      {"input", 0},
      {"filter", 1},
      {"bias", 2},
    }},
    {"softplus", {
      {"x", 0},
    }},
    {"softshrink", {
      {"x", 0},
    }},
    {"softsign", {
      {"x", 0},
    }},
    {"solve", {
      {"x", 0},
      {"y", 1},
    }},
    {"sparse_attention", {
      {"q", 0},
      {"k", 1},
      {"v", 2},
      {"offset", 3},
      {"columns", 4},
      {"key_padding_mask", 5},
      {"attn_mask", 6},
    }},
    {"spectral_norm", {
      {"weight", 0},
      {"u", 1},
      {"v", 2},
    }},
    {"split", {
      {"x", 0},
    }},
    {"split_with_num", {
      {"x", 0},
    }},
    {"sqrt", {
      {"x", 0},
    }},
    {"square", {
      {"x", 0},
    }},
    {"squared_l2_norm", {
      {"x", 0},
    }},
    {"squeeze", {
      {"x", 0},
    }},
    {"stack", {
      {"x", 0},
    }},
    {"standard_gamma", {
      {"x", 0},
    }},
    {"stanh", {
      {"x", 0},
    }},
    {"std", {
      {"x", 0},
    }},
    {"stft", {
      {"x", 0},
      {"window", 1},
    }},
    {"strided_slice", {
      {"x", 0},
    }},
    {"sum", {
      {"x", 0},
    }},
    {"svd", {
      {"x", 0},
    }},
    {"svdvals", {
      {"x", 0},
    }},
    {"swiglu", {
      {"x", 0},
      {"y", 1},
    }},
    {"swish", {
      {"x", 0},
    }},
    {"sync_batch_norm_", {
      {"x", 0},
      {"mean", 1},
      {"variance", 2},
      {"scale", 3},
      {"bias", 4},
    }},
    {"sync_calc_stream", {
      {"x", 0},
    }},
    {"take_along_axis", {
      {"arr", 0},
      {"indices", 1},
    }},
    {"tan", {
      {"x", 0},
    }},
    {"tanh", {
      {"x", 0},
    }},
    {"tanh_shrink", {
      {"x", 0},
    }},
    {"tdm_child", {
      {"x", 0},
      {"tree_info", 1},
    }},
    {"tdm_sampler", {
      {"x", 0},
      {"travel", 1},
      {"layer", 2},
    }},
    {"temporal_shift", {
      {"x", 0},
    }},
    {"thresholded_relu", {
      {"x", 0},
    }},
    {"top_p_sampling", {
      {"x", 0},
      {"ps", 1},
      {"threshold", 2},
      {"topp_seed", 3},
    }},
    {"topk", {
      {"x", 0},
    }},
    {"trace", {
      {"x", 0},
    }},
    {"trans_layout", {
      {"x", 0},
    }},
    {"transpose", {
      {"x", 0},
    }},
    {"triangular_solve", {
      {"x", 0},
      {"y", 1},
    }},
    {"tril", {
      {"x", 0},
    }},
    {"tril_indices", {
      {"rows", 0},
      {"cols", 1},
      {"offset", 2},
      {"dtype", 3},
      {"place", 4},
    }},
    {"trilinear_interp", {
      {"x", 0},
      {"out_size", 1},
      {"size_tensor", 2},
      {"scale_tensor", 3},
    }},
    {"triu", {
      {"x", 0},
    }},
    {"triu_indices", {
      {"row", 0},
      {"col", 1},
      {"offset", 2},
      {"dtype", 3},
      {"place", 4},
    }},
    {"trunc", {
      {"input", 0},
    }},
    {"trunc_divide", {
      {"x", 0},
      {"y", 1},
    }},
    {"truncated_gaussian_random", {
      {"shape", 0},
      {"mean", 1},
      {"std", 2},
      {"seed", 3},
      {"a", 4},
      {"b", 5},
      {"dtype", 6},
      {"place", 7},
    }},
    {"unbind", {
      {"input", 0},
    }},
    {"unfold", {
      {"x", 0},
    }},
    {"uniform", {
      {"shape", 0},
      {"dtype", 1},
      {"min", 2},
      {"max", 3},
      {"seed", 4},
      {"place", 5},
    }},
    {"uniform_inplace", {
      {"x", 0},
    }},
    {"uniform_random_batch_size_like", {
      {"input", 0},
    }},
    {"unique_consecutive", {
      {"x", 0},
    }},
    {"unpool", {
      {"x", 0},
      {"indices", 1},
    }},
    {"unpool3d", {
      {"x", 0},
      {"indices", 1},
    }},
    {"unsqueeze", {
      {"x", 0},
    }},
    {"unstack", {
      {"x", 0},
    }},
    {"update_loss_scaling_", {
      {"x", 0},
      {"found_infinite", 1},
      {"prev_loss_scaling", 2},
      {"in_good_steps", 3},
      {"in_bad_steps", 4},
    }},
    {"var", {
      {"x", 0},
    }},
    {"variance", {
      {"x", 0},
    }},
    {"view_dtype", {
      {"input", 0},
    }},
    {"view_shape", {
      {"input", 0},
    }},
    {"view_slice", {
      {"input", 0},
    }},
    {"viterbi_decode", {
      {"potentials", 0},
      {"transition_params", 1},
      {"lengths", 2},
    }},
    {"warpctc", {
      {"logits", 0},
      {"label", 1},
      {"logits_length", 2},
      {"labels_length", 3},
    }},
    {"warprnnt", {
      {"input", 0},
      {"label", 1},
      {"input_lengths", 2},
      {"label_lengths", 3},
    }},
    {"weight_dequantize", {
      {"x", 0},
      {"scale", 1},
    }},
    {"weight_only_linear", {
      {"x", 0},
      {"weight", 1},
      {"bias", 2},
      {"weight_scale", 3},
    }},
    {"weight_quantize", {
      {"x", 0},
    }},
    {"weighted_sample_neighbors", {
      {"row", 0},
      {"colptr", 1},
      {"edge_weight", 2},
      {"input_nodes", 3},
      {"eids", 4},
    }},
    {"where", {
      {"condition", 0},
      {"x", 1},
      {"y", 2},
    }},
    {"yolo_box", {
      {"x", 0},
      {"img_size", 1},
    }},
    {"yolo_box_head", {
      {"x", 0},
    }},
    {"yolo_box_post", {
      {"boxes0", 0},
      {"boxes1", 1},
      {"boxes2", 2},
      {"image_shape", 3},
      {"image_scale", 4},
    }},
    {"yolo_loss", {
      {"x", 0},
      {"gt_box", 1},
      {"gt_label", 2},
      {"gt_score", 3},
    }},
    {"zeros", {
      {"shape", 0},
      {"dtype", 1},
      {"place", 2},
    }},
    {"zeros_like", {
      {"x", 0},
    }},
  };
  static const std::unordered_map<std::string, int> empty;
  auto it = all.find(op_name);
  return it != all.end() ? it->second : empty;
}

// Output name → positional index
inline const std::unordered_map<std::string, int>&
GetOpOutputIndices_v3.4(const std::string& op_name) {
  static const std::unordered_map<std::string, std::unordered_map<std::string, int>> all = {
    {"abs", {
      {"out", 0},
    }},
    {"accuracy", {
      {"accuracy", 0},
      {"correct", 1},
      {"total", 2},
    }},
    {"accuracy_check", {
      {"out", 0},
    }},
    {"acos", {
      {"out", 0},
    }},
    {"acosh", {
      {"out", 0},
    }},
    {"adadelta_", {
      {"param_out", 0},
      {"moment_out", 1},
      {"inf_norm_out", 2},
      {"master_param_out", 3},
    }},
    {"adagrad_", {
      {"param_out", 0},
      {"moment_out", 1},
      {"master_param_out", 2},
    }},
    {"adam_", {
      {"param_out", 0},
      {"moment1_out", 1},
      {"moment2_out", 2},
      {"moment2_max_out", 3},
      {"beta1_pow_out", 4},
      {"beta2_pow_out", 5},
      {"master_param_out", 6},
    }},
    {"adamax_", {
      {"param_out", 0},
      {"moment_out", 1},
      {"inf_norm_out", 2},
      {"master_param_out", 3},
    }},
    {"adamw_", {
      {"param_out", 0},
      {"moment1_out", 1},
      {"moment2_out", 2},
      {"moment2_max_out", 3},
      {"beta1_pow_out", 4},
      {"beta2_pow_out", 5},
      {"master_param_out", 6},
    }},
    {"addmm", {
      {"out", 0},
    }},
    {"affine_grid", {
      {"output", 0},
    }},
    {"all", {
      {"out", 0},
    }},
    {"all_gather", {
      {"out", 0},
    }},
    {"all_reduce", {
      {"out", 0},
    }},
    {"all_to_all", {
      {"out", 0},
    }},
    {"allclose", {
      {"out", 0},
    }},
    {"amax", {
      {"out", 0},
    }},
    {"amin", {
      {"out", 0},
    }},
    {"aminmax", {
      {"min", 0},
      {"max", 1},
    }},
    {"any", {
      {"out", 0},
    }},
    {"ap_trivial_fusion_begin", {
      {"out", 0},
    }},
    {"ap_trivial_fusion_end", {
      {"out", 0},
    }},
    {"apply_per_channel_scale", {
      {"out", 0},
    }},
    {"argmax", {
      {"out", 0},
    }},
    {"argmin", {
      {"out", 0},
    }},
    {"argsort", {
      {"out", 0},
      {"indices", 1},
    }},
    {"asgd_", {
      {"param_out", 0},
      {"d_out", 1},
      {"y_out", 2},
      {"master_param_out", 3},
    }},
    {"asin", {
      {"out", 0},
    }},
    {"asinh", {
      {"out", 0},
    }},
    {"assign_out_", {
      {"out", 0},
    }},
    {"assign_pos", {
      {"out", 0},
    }},
    {"assign_value_", {
      {"out", 0},
    }},
    {"atan", {
      {"out", 0},
    }},
    {"atan2", {
      {"out", 0},
    }},
    {"atanh", {
      {"out", 0},
    }},
    {"auc", {
      {"auc", 0},
      {"stat_pos_out", 1},
      {"stat_neg_out", 2},
    }},
    {"average_accumulates_", {
      {"out_sum_1", 0},
      {"out_sum_2", 1},
      {"out_sum_3", 2},
      {"out_num_accumulates", 3},
      {"out_old_num_accumulates", 4},
      {"out_num_updates", 5},
    }},
    {"baddbmm", {
      {"out", 0},
    }},
    {"barrier", {
      {"out", 0},
    }},
    {"batch_fc", {
      {"out", 0},
    }},
    {"batched_gemm", {
      {"output", 0},
    }},
    {"bce_loss", {
      {"out", 0},
    }},
    {"bernoulli", {
      {"out", 0},
    }},
    {"bicubic_interp", {
      {"output", 0},
    }},
    {"bilinear_interp", {
      {"output", 0},
    }},
    {"bincount", {
      {"out", 0},
    }},
    {"binomial", {
      {"out", 0},
    }},
    {"bitwise_and", {
      {"out", 0},
    }},
    {"bitwise_left_shift", {
      {"out", 0},
    }},
    {"bitwise_not", {
      {"out", 0},
    }},
    {"bitwise_or", {
      {"out", 0},
    }},
    {"bitwise_right_shift", {
      {"out", 0},
    }},
    {"bitwise_xor", {
      {"out", 0},
    }},
    {"bmm", {
      {"out", 0},
    }},
    {"box_coder", {
      {"output_box", 0},
    }},
    {"broadcast", {
      {"out", 0},
    }},
    {"build_src_rank_and_local_expert_id", {
      {"vector", 0},
      {"local_expert_id", 1},
    }},
    {"c_allreduce_sum", {
      {"out", 0},
    }},
    {"c_concat", {
      {"out", 0},
    }},
    {"c_identity", {
      {"out", 0},
    }},
    {"c_scatter", {
      {"out", 0},
    }},
    {"c_softmax_with_cross_entropy", {
      {"softmax", 0},
      {"loss", 1},
    }},
    {"c_split", {
      {"out", 0},
    }},
    {"cal_aux_loss", {
      {"l_aux_loss", 0},
      {"seqlen_float", 1},
      {"ce", 2},
    }},
    {"calc_reduced_attn_scores", {
      {"reduced_scores", 0},
    }},
    {"cast", {
      {"out", 0},
    }},
    {"ceil", {
      {"out", 0},
    }},
    {"celu", {
      {"out", 0},
    }},
    {"channel_shuffle", {
      {"out", 0},
    }},
    {"check_finite_and_unscale_", {
      {"found_infinite", 0},
    }},
    {"check_numerics", {
      {"stats", 0},
      {"values", 1},
    }},
    {"class_center_sample", {
      {"remapped_label", 0},
      {"sampled_local_class_center", 1},
    }},
    {"clip", {
      {"out", 0},
    }},
    {"clip_by_norm", {
      {"out", 0},
    }},
    {"coalesce_tensor", {
      {"fused_output", 0},
    }},
    {"conv2d_transpose", {
      {"out", 0},
    }},
    {"conv2d_transpose_bias", {
      {"out", 0},
    }},
    {"conv3d_transpose", {
      {"out", 0},
    }},
    {"copy_to", {
      {"out", 0},
    }},
    {"copysign", {
      {"out", 0},
    }},
    {"correlation", {
      {"out", 0},
    }},
    {"cos", {
      {"out", 0},
    }},
    {"cosh", {
      {"out", 0},
    }},
    {"crop", {
      {"out", 0},
    }},
    {"cross_entropy_with_softmax", {
      {"softmax", 0},
      {"loss", 1},
    }},
    {"cross_entropy_with_softmax_bwd_w_downcast", {
      {"input_grad", 0},
    }},
    {"cummax", {
      {"out", 0},
      {"indices", 1},
    }},
    {"cummin", {
      {"out", 0},
      {"indices", 1},
    }},
    {"cumprod", {
      {"out", 0},
    }},
    {"cumsum", {
      {"out", 0},
    }},
    {"data", {
      {"out", 0},
    }},
    {"decayed_adagrad", {
      {"param_out", 0},
      {"moment_out", 1},
    }},
    {"decode_jpeg", {
      {"out", 0},
    }},
    {"deformable_conv", {
      {"out", 0},
    }},
    {"depthwise_conv2d", {
      {"out", 0},
    }},
    {"depthwise_conv2d_bias", {
      {"out", 0},
    }},
    {"depthwise_conv2d_transpose", {
      {"out", 0},
    }},
    {"depthwise_conv3d_bias", {
      {"out", 0},
    }},
    {"dequantize_abs_max", {
      {"out", 0},
    }},
    {"dequantize_log", {
      {"out", 0},
    }},
    {"dgc", {
      {"u_out", 0},
      {"v_out", 1},
      {"encode_grad", 2},
      {"grad_out", 3},
      {"k", 4},
      {"gather_buff", 5},
    }},
    {"dgc_clip_by_norm", {
      {"out", 0},
    }},
    {"diag", {
      {"out", 0},
    }},
    {"diag_embed", {
      {"out", 0},
    }},
    {"digamma", {
      {"out", 0},
    }},
    {"dirichlet", {
      {"out", 0},
    }},
    {"disable_check_model_nan_inf", {
      {"out", 0},
    }},
    {"dpsgd", {
      {"param_out", 0},
    }},
    {"dropout", {
      {"out", 0},
      {"mask", 1},
    }},
    {"edit_distance", {
      {"sequencenum", 0},
      {"out", 1},
    }},
    {"eig", {
      {"out_w", 0},
      {"out_v", 1},
    }},
    {"eigh", {
      {"out_w", 0},
      {"out_v", 1},
    }},
    {"eigvals", {
      {"out", 0},
    }},
    {"eigvalsh", {
      {"eigenvalues", 0},
      {"eigenvectors", 1},
    }},
    {"elu", {
      {"out", 0},
    }},
    {"embedding_grad_add_to", {
      {"main_grad_out", 0},
    }},
    {"empty", {
      {"out", 0},
    }},
    {"empty_like", {
      {"out", 0},
    }},
    {"enable_check_model_nan_inf", {
      {"out", 0},
    }},
    {"equal_all", {
      {"out", 0},
    }},
    {"erf", {
      {"out", 0},
    }},
    {"erfinv", {
      {"out", 0},
    }},
    {"exp", {
      {"out", 0},
    }},
    {"expand", {
      {"out", 0},
    }},
    {"expand_as", {
      {"out", 0},
    }},
    {"expand_modality_expert_id", {
      {"expert_id_out", 0},
    }},
    {"expm1", {
      {"out", 0},
    }},
    {"exponential_", {
      {"out", 0},
    }},
    {"eye", {
      {"out", 0},
    }},
    {"fake_channel_wise_dequantize_max_abs", {
      {"out", 0},
    }},
    {"fake_channel_wise_quantize_abs_max", {
      {"out", 0},
      {"out_scale", 1},
    }},
    {"fake_channel_wise_quantize_dequantize_abs_max", {
      {"out", 0},
      {"out_scale", 1},
    }},
    {"fake_dequantize_max_abs", {
      {"out", 0},
    }},
    {"fake_quantize_abs_max", {
      {"out", 0},
      {"out_scale", 1},
    }},
    {"fake_quantize_dequantize_abs_max", {
      {"out", 0},
      {"out_scale", 1},
    }},
    {"fake_quantize_dequantize_moving_average_abs_max", {
      {"out", 0},
      {"out_scale", 1},
      {"out_state", 2},
      {"out_accum", 3},
    }},
    {"fake_quantize_moving_average_abs_max", {
      {"out", 0},
      {"out_scale", 1},
      {"out_state", 2},
      {"out_accum", 3},
    }},
    {"fake_quantize_range_abs_max", {
      {"out", 0},
      {"out_scale", 1},
      {"out_scales", 2},
    }},
    {"fast_ln", {
      {"y", 0},
      {"mean", 1},
      {"invvar", 2},
    }},
    {"fast_rms_norm", {
      {"y", 0},
      {"invvar", 1},
    }},
    {"fill", {
      {"out", 0},
    }},
    {"fill_diagonal", {
      {"out", 0},
    }},
    {"fill_diagonal_tensor", {
      {"out", 0},
    }},
    {"flash_attn", {
      {"out", 0},
      {"softmax", 1},
      {"softmax_lse", 2},
      {"seed_offset", 3},
    }},
    {"flash_attn_qkvpacked", {
      {"out", 0},
      {"softmax", 1},
      {"softmax_lse", 2},
      {"seed_offset", 3},
    }},
    {"flash_attn_unpadded", {
      {"out", 0},
      {"softmax", 1},
      {"softmax_lse", 2},
      {"seed_offset", 3},
    }},
    {"flash_attn_v3", {
      {"out", 0},
      {"softmax_lse", 1},
    }},
    {"flash_attn_v3_varlen", {
      {"out", 0},
      {"softmax_lse", 1},
    }},
    {"flash_attn_varlen_qkvpacked", {
      {"out", 0},
      {"softmax", 1},
      {"softmax_lse", 2},
      {"seed_offset", 3},
    }},
    {"flashmask_attention", {
      {"out", 0},
      {"softmax", 1},
      {"softmax_lse", 2},
      {"seed_offset", 3},
    }},
    {"flashmask_attention_v2", {
      {"out", 0},
      {"softmax_lse", 1},
    }},
    {"flashmask_get_unique_id", {
      {"out", 0},
    }},
    {"flatten", {
      {"out", 0},
    }},
    {"floor", {
      {"out", 0},
    }},
    {"fmax", {
      {"out", 0},
    }},
    {"fmin", {
      {"out", 0},
    }},
    {"fold", {
      {"out", 0},
    }},
    {"fp8_quant_blockwise", {
      {"out", 0},
      {"scale", 1},
      {"out_transposed", 2},
      {"scale_transposed", 3},
    }},
    {"fractional_max_pool2d", {
      {"out", 0},
      {"mask", 1},
    }},
    {"fractional_max_pool3d", {
      {"out", 0},
      {"mask", 1},
    }},
    {"frame", {
      {"out", 0},
    }},
    {"frobenius_norm", {
      {"out", 0},
    }},
    {"ftrl", {
      {"param_out", 0},
      {"squared_accum_out", 1},
      {"linear_accum_out", 2},
    }},
    {"full", {
      {"out", 0},
    }},
    {"full_", {
      {"out", 0},
    }},
    {"full_batch_size_like", {
      {"out", 0},
    }},
    {"full_int_array", {
      {"out", 0},
    }},
    {"full_like", {
      {"out", 0},
    }},
    {"full_with_tensor", {
      {"out", 0},
    }},
    {"fused_batch_norm_act", {
      {"out", 0},
      {"mean_out", 1},
      {"variance_out", 2},
      {"saved_mean", 3},
      {"saved_variance", 4},
      {"reserve_space", 5},
    }},
    {"fused_bn_add_activation", {
      {"out", 0},
      {"mean_out", 1},
      {"variance_out", 2},
      {"saved_mean", 3},
      {"saved_variance", 4},
      {"reserve_space", 5},
    }},
    {"fused_rms_norm_ext", {
      {"y", 0},
      {"invvar", 1},
    }},
    {"fused_rms_norm_quant", {
      {"out", 0},
      {"residual_out", 1},
      {"inv_var", 2},
    }},
    {"fused_softmax_mask", {
      {"out", 0},
    }},
    {"fused_softmax_mask_upper_triangle", {
      {"Out", 0},
    }},
    {"gammaincc", {
      {"out", 0},
    }},
    {"gammaln", {
      {"out", 0},
    }},
    {"gather", {
      {"out", 0},
    }},
    {"gather_nd", {
      {"out", 0},
    }},
    {"gather_tree", {
      {"out", 0},
    }},
    {"gaussian", {
      {"out", 0},
    }},
    {"gaussian_inplace", {
      {"out", 0},
    }},
    {"gelu", {
      {"out", 0},
    }},
    {"generate_proposals", {
      {"rpn_rois", 0},
      {"rpn_roi_probs", 1},
      {"rpn_rois_num", 2},
    }},
    {"global_gather", {
      {"out", 0},
    }},
    {"global_scatter", {
      {"out", 0},
    }},
    {"graph_khop_sampler", {
      {"out_src", 0},
      {"out_dst", 1},
      {"sample_index", 2},
      {"reindex_x", 3},
      {"out_eids", 4},
    }},
    {"graph_sample_neighbors", {
      {"out", 0},
      {"out_count", 1},
      {"out_eids", 2},
    }},
    {"grid_sample", {
      {"out", 0},
    }},
    {"group_norm", {
      {"y", 0},
      {"mean", 1},
      {"variance", 2},
    }},
    {"hardtanh", {
      {"out", 0},
    }},
    {"heaviside", {
      {"out", 0},
    }},
    {"histogram", {
      {"out", 0},
    }},
    {"hsigmoid_loss", {
      {"out", 0},
      {"pre_out", 1},
      {"w_out", 2},
    }},
    {"huber_loss", {
      {"out", 0},
      {"residual", 1},
    }},
    {"i0", {
      {"out", 0},
    }},
    {"i0e", {
      {"out", 0},
    }},
    {"i1", {
      {"out", 0},
    }},
    {"i1e", {
      {"out", 0},
    }},
    {"identity_loss", {
      {"out", 0},
    }},
    {"increment", {
      {"out", 0},
    }},
    {"index_add", {
      {"out", 0},
    }},
    {"index_fill", {
      {"out", 0},
    }},
    {"index_put", {
      {"out", 0},
    }},
    {"index_select", {
      {"out", 0},
    }},
    {"index_select_strided", {
      {"out", 0},
    }},
    {"instance_norm", {
      {"y", 0},
      {"saved_mean", 1},
      {"saved_variance", 2},
    }},
    {"int_bincount", {
      {"out", 0},
    }},
    {"interp_antialias", {
      {"output", 0},
    }},
    {"inverse", {
      {"out", 0},
    }},
    {"is_empty", {
      {"out", 0},
    }},
    {"isclose", {
      {"out", 0},
    }},
    {"isfinite", {
      {"out", 0},
    }},
    {"isinf", {
      {"out", 0},
    }},
    {"isnan", {
      {"out", 0},
    }},
    {"kldiv_loss", {
      {"out", 0},
    }},
    {"kthvalue", {
      {"out", 0},
      {"indices", 1},
    }},
    {"l1_norm", {
      {"out", 0},
    }},
    {"lamb_", {
      {"param_out", 0},
      {"moment1_out", 1},
      {"moment2_out", 2},
      {"beta1_pow_out", 3},
      {"beta2_pow_out", 4},
      {"master_param_outs", 5},
    }},
    {"layer_norm", {
      {"out", 0},
      {"mean", 1},
      {"variance", 2},
    }},
    {"leaky_relu", {
      {"out", 0},
    }},
    {"lerp", {
      {"out", 0},
    }},
    {"lgamma", {
      {"out", 0},
    }},
    {"limit_by_capacity", {
      {"out", 0},
    }},
    {"linear_interp", {
      {"output", 0},
    }},
    {"linear_v2", {
      {"out", 0},
    }},
    {"linspace", {
      {"out", 0},
    }},
    {"llm_int8_linear", {
      {"out", 0},
    }},
    {"log", {
      {"out", 0},
    }},
    {"log10", {
      {"out", 0},
    }},
    {"log1p", {
      {"out", 0},
    }},
    {"log2", {
      {"out", 0},
    }},
    {"log_softmax", {
      {"out", 0},
    }},
    {"logcumsumexp", {
      {"out", 0},
    }},
    {"logical_and", {
      {"out", 0},
    }},
    {"logical_not", {
      {"out", 0},
    }},
    {"logical_or", {
      {"out", 0},
    }},
    {"logical_xor", {
      {"out", 0},
    }},
    {"logit", {
      {"out", 0},
    }},
    {"logspace", {
      {"out", 0},
    }},
    {"logsumexp", {
      {"out", 0},
    }},
    {"lp_pool2d", {
      {"out", 0},
    }},
    {"lstsq", {
      {"solution", 0},
      {"residuals", 1},
      {"rank", 2},
      {"singular_values", 3},
    }},
    {"lu", {
      {"out", 0},
      {"pivots", 1},
      {"infos", 2},
    }},
    {"lu_solve", {
      {"out", 0},
    }},
    {"lu_unpack", {
      {"pmat", 0},
      {"l", 1},
      {"u", 2},
    }},
    {"margin_cross_entropy", {
      {"softmax", 0},
      {"loss", 1},
    }},
    {"masked_multihead_attention_", {
      {"out", 0},
      {"cache_kv_out", 1},
      {"beam_cache_offset_out", 2},
    }},
    {"matrix_nms", {
      {"out", 0},
      {"index", 1},
      {"roisnum", 2},
    }},
    {"matrix_rank", {
      {"out", 0},
    }},
    {"matrix_rank_atol_rtol", {
      {"out", 0},
    }},
    {"matrix_rank_tol", {
      {"out", 0},
    }},
    {"max", {
      {"out", 0},
    }},
    {"max_pool2d_with_index", {
      {"out", 0},
      {"mask", 1},
    }},
    {"max_pool3d_with_index", {
      {"out", 0},
      {"mask", 1},
    }},
    {"max_with_index", {
      {"values", 0},
      {"indices", 1},
    }},
    {"maxout", {
      {"out", 0},
    }},
    {"mean", {
      {"out", 0},
    }},
    {"median", {
      {"out", 0},
      {"medians", 1},
    }},
    {"memory_efficient_attention", {
      {"output", 0},
      {"logsumexp", 1},
      {"seed_and_offset", 2},
    }},
    {"merge_selected_rows", {
      {"out", 0},
    }},
    {"min_with_index", {
      {"values", 0},
      {"indices", 1},
    }},
    {"mode", {
      {"out", 0},
      {"indices", 1},
    }},
    {"moe_combine", {
      {"y", 0},
    }},
    {"moe_combine_auto", {
      {"y", 0},
    }},
    {"moe_combine_no_weight", {
      {"y", 0},
    }},
    {"moe_gate_dispatch", {
      {"y", 0},
      {"combine_weights", 1},
      {"scatter_index", 2},
      {"expert_offset", 3},
      {"expert_id", 4},
    }},
    {"moe_gate_dispatch_and_quant", {
      {"out_fp8", 0},
      {"scale", 1},
      {"combine_weights", 2},
      {"scatter_index", 3},
      {"expert_offset", 4},
      {"expert_id", 5},
    }},
    {"moe_gate_dispatch_auto", {
      {"y", 0},
      {"combine_weights", 1},
      {"scatter_index", 2},
      {"expert_offset", 3},
      {"expert_id", 4},
    }},
    {"moe_gate_dispatch_partial_nosoftmaxtopk", {
      {"y", 0},
      {"combine_weights_out", 1},
      {"scatter_index", 2},
      {"scatter_index_rev", 3},
      {"expert_offset", 4},
      {"expert_nums_local", 5},
    }},
    {"moe_gate_dispatch_permute", {
      {"y", 0},
      {"combine_weights", 1},
      {"scatter_index", 2},
      {"expert_offset", 3},
      {"expert_id", 4},
    }},
    {"moe_permute", {
      {"hidden_states_unzipped", 0},
      {"zipped_expertwise_rowmap", 1},
      {"token_prob_unzipped", 2},
      {"scale_unzipped", 3},
      {"expert_indices", 4},
    }},
    {"moe_unpermute", {
      {"hidden_states", 0},
      {"expert_prob_topk", 1},
    }},
    {"momentum_", {
      {"param_out", 0},
      {"velocity_out", 1},
      {"master_param_out", 2},
    }},
    {"mp_allreduce_sum", {
      {"out", 0},
    }},
    {"multiclass_nms3", {
      {"out", 0},
      {"index", 1},
      {"nms_rois_num", 2},
    }},
    {"multinomial", {
      {"out", 0},
    }},
    {"nadam_", {
      {"param_out", 0},
      {"momentum_decay_pow_out", 1},
      {"beta2_pow_out", 2},
      {"mu_product_out", 3},
      {"moment1_out", 4},
      {"moment2_out", 5},
      {"master_param_out", 6},
    }},
    {"nanmedian", {
      {"out", 0},
      {"medians", 1},
    }},
    {"nansum", {
      {"out", 0},
    }},
    {"nearest_interp", {
      {"output", 0},
    }},
    {"nextafter", {
      {"out", 0},
    }},
    {"nll_loss", {
      {"out", 0},
      {"total_weight", 1},
    }},
    {"nms", {
      {"out", 0},
    }},
    {"nonzero", {
      {"out", 0},
    }},
    {"norm", {
      {"out", 0},
      {"norm", 1},
    }},
    {"number_count", {
      {"out", 0},
    }},
    {"numel", {
      {"size", 0},
    }},
    {"one_hot", {
      {"out", 0},
    }},
    {"ones", {
      {"out", 0},
    }},
    {"ones_like", {
      {"out", 0},
    }},
    {"p_norm", {
      {"out", 0},
    }},
    {"pad3d", {
      {"out", 0},
    }},
    {"partial_allgather", {
      {"out", 0},
    }},
    {"partial_concat", {
      {"out", 0},
    }},
    {"partial_sum", {
      {"out", 0},
    }},
    {"polygamma", {
      {"out", 0},
    }},
    {"pool2d", {
      {"out", 0},
    }},
    {"pool3d", {
      {"out", 0},
    }},
    {"pow", {
      {"out", 0},
    }},
    {"prelu", {
      {"out", 0},
    }},
    {"prior_box", {
      {"out", 0},
      {"var", 1},
    }},
    {"prune_gate_by_capacity", {
      {"out_gate_idx", 0},
    }},
    {"put_along_axis", {
      {"out", 0},
    }},
    {"qr", {
      {"q", 0},
      {"r", 1},
    }},
    {"radam_", {
      {"param_out", 0},
      {"beta1_pow_out", 1},
      {"beta2_pow_out", 2},
      {"rho_out", 3},
      {"moment1_out", 4},
      {"moment2_out", 5},
      {"master_param_out", 6},
    }},
    {"randint", {
      {"out", 0},
    }},
    {"random", {
      {"out", 0},
    }},
    {"random_routing", {
      {"out", 0},
    }},
    {"randperm", {
      {"out", 0},
    }},
    {"rank_attention", {
      {"input_help", 0},
      {"out", 1},
      {"ins_rank", 2},
    }},
    {"read_file", {
      {"out", 0},
    }},
    {"reciprocal", {
      {"out", 0},
    }},
    {"reduce", {
      {"out", 0},
    }},
    {"reduce_as", {
      {"out", 0},
    }},
    {"reduce_scatter", {
      {"out", 0},
    }},
    {"reindex_graph", {
      {"reindex_src", 0},
      {"reindex_dst", 1},
      {"out_nodes", 2},
    }},
    {"relu", {
      {"out", 0},
    }},
    {"renorm", {
      {"out", 0},
    }},
    {"repeat_interleave", {
      {"out", 0},
    }},
    {"repeat_interleave_with_tensor_index", {
      {"out", 0},
    }},
    {"reshape", {
      {"out", 0},
    }},
    {"rint", {
      {"out", 0},
    }},
    {"rms_norm", {
      {"y", 0},
      {"invvar", 1},
    }},
    {"rmsprop_", {
      {"param_out", 0},
      {"moment_out", 1},
      {"mean_square_out", 2},
      {"mean_grad_out", 3},
      {"master_param_outs", 4},
    }},
    {"rnn", {
      {"out", 0},
      {"dropout_state_out", 1},
      {"reserve", 2},
    }},
    {"roi_pool", {
      {"out", 0},
      {"arg_max", 1},
    }},
    {"roll", {
      {"out", 0},
    }},
    {"round", {
      {"out", 0},
    }},
    {"rprop_", {
      {"param_out", 0},
      {"prev_out", 1},
      {"learning_rate_out", 2},
      {"master_param_out", 3},
    }},
    {"rrelu", {
      {"out", 0},
      {"noise", 1},
    }},
    {"rsqrt", {
      {"out", 0},
    }},
    {"scale", {
      {"out", 0},
    }},
    {"scatter", {
      {"out", 0},
    }},
    {"searchsorted", {
      {"out", 0},
    }},
    {"segment_pool", {
      {"out", 0},
      {"summed_ids", 1},
    }},
    {"send_u_recv", {
      {"out", 0},
      {"dst_count", 1},
    }},
    {"send_ue_recv", {
      {"out", 0},
      {"dst_count", 1},
    }},
    {"send_uv", {
      {"out", 0},
    }},
    {"sequence_mask", {
      {"y", 0},
    }},
    {"set_value_with_tensor", {
      {"out", 0},
    }},
    {"sgd_", {
      {"param_out", 0},
      {"master_param_out", 1},
    }},
    {"shape", {
      {"out", 0},
    }},
    {"shape64", {
      {"out", 0},
    }},
    {"shard_index", {
      {"out", 0},
    }},
    {"shuffle_batch", {
      {"out", 0},
      {"shuffle_idx", 1},
      {"seed_out", 2},
    }},
    {"shuffle_channel", {
      {"out", 0},
    }},
    {"sign", {
      {"out", 0},
    }},
    {"silu", {
      {"out", 0},
    }},
    {"sin", {
      {"out", 0},
    }},
    {"sinh", {
      {"out", 0},
    }},
    {"slogdet_v2", {
      {"sign", 0},
      {"logdet", 1},
    }},
    {"sqrt", {
      {"out", 0},
    }},
    {"squared_l2_norm", {
      {"out", 0},
    }},
    {"squeeze", {
      {"out", 0},
    }},
    {"standard_gamma", {
      {"out", 0},
    }},
    {"stanh", {
      {"out", 0},
    }},
    {"std", {
      {"out", 0},
    }},
    {"sum", {
      {"out", 0},
    }},
    {"svd", {
      {"u", 0},
      {"s", 1},
      {"vh", 2},
    }},
    {"svdvals", {
      {"s", 0},
    }},
    {"swiglu", {
      {"out", 0},
    }},
    {"swish", {
      {"out", 0},
    }},
    {"sync_batch_norm_", {
      {"out", 0},
      {"mean_out", 1},
      {"variance_out", 2},
      {"saved_mean", 3},
      {"saved_variance", 4},
      {"reserve_space", 5},
    }},
    {"sync_calc_stream", {
      {"out", 0},
    }},
    {"tan", {
      {"out", 0},
    }},
    {"tanh", {
      {"out", 0},
    }},
    {"tdm_sampler", {
      {"out", 0},
      {"labels", 1},
      {"mask", 2},
    }},
    {"temporal_shift", {
      {"out", 0},
    }},
    {"thresholded_relu", {
      {"out", 0},
    }},
    {"top_p_sampling", {
      {"ids", 0},
      {"topk_scores", 1},
      {"topk_ids", 2},
    }},
    {"topk", {
      {"out", 0},
      {"indices", 1},
    }},
    {"transpose", {
      {"out", 0},
    }},
    {"tril", {
      {"out", 0},
    }},
    {"tril_indices", {
      {"out", 0},
    }},
    {"trilinear_interp", {
      {"output", 0},
    }},
    {"triu", {
      {"out", 0},
    }},
    {"triu_indices", {
      {"out", 0},
    }},
    {"trunc", {
      {"out", 0},
    }},
    {"trunc_divide", {
      {"out", 0},
    }},
    {"truncated_gaussian_random", {
      {"out", 0},
    }},
    {"unfold", {
      {"out", 0},
    }},
    {"uniform", {
      {"out", 0},
    }},
    {"uniform_inplace", {
      {"out", 0},
    }},
    {"unique_consecutive", {
      {"out", 0},
      {"index", 1},
      {"counts", 2},
    }},
    {"unpool", {
      {"out", 0},
    }},
    {"unpool3d", {
      {"out", 0},
    }},
    {"unsqueeze", {
      {"out", 0},
    }},
    {"update_loss_scaling_", {
      {"loss_scaling", 0},
      {"out_good_steps", 1},
      {"out_bad_steps", 2},
    }},
    {"var", {
      {"out", 0},
    }},
    {"variance", {
      {"out", 0},
    }},
    {"view_dtype", {
      {"out", 0},
    }},
    {"view_shape", {
      {"out", 0},
    }},
    {"viterbi_decode", {
      {"scores", 0},
      {"path", 1},
    }},
    {"warpctc", {
      {"loss", 0},
      {"warpctcgrad", 1},
    }},
    {"warprnnt", {
      {"loss", 0},
      {"warprnntgrad", 1},
    }},
    {"weight_dequantize", {
      {"out", 0},
    }},
    {"weight_only_linear", {
      {"out", 0},
    }},
    {"weight_quantize", {
      {"out", 0},
      {"scale", 1},
    }},
    {"weighted_sample_neighbors", {
      {"out_neighbors", 0},
      {"out_count", 1},
      {"out_eids", 2},
    }},
    {"where", {
      {"out", 0},
    }},
    {"yolo_box", {
      {"boxes", 0},
      {"scores", 1},
    }},
    {"yolo_box_head", {
      {"out", 0},
    }},
    {"yolo_box_post", {
      {"out", 0},
      {"nms_rois_num", 1},
    }},
    {"yolo_loss", {
      {"loss", 0},
      {"objectness_mask", 1},
      {"gt_match_mask", 2},
    }},
    {"zeros", {
      {"out", 0},
    }},
    {"zeros_like", {
      {"out", 0},
    }},
  };
  static const std::unordered_map<std::string, int> empty;
  auto it = all.find(op_name);
  return it != all.end() ? it->second : empty;
}

// Op arg name mappings: phi_arg_name → fluid/legacy_arg_name
inline const std::unordered_map<std::string, std::string>&
GetOpArgMappings_v3.4(const std::string& op_name) {
  static const std::unordered_map<std::string, std::unordered_map<std::string, std::string>> all = {
    {"abs", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"accuracy", {
      {"accuracy", "Accuracy"},
      {"correct", "Correct"},
      {"indices", "Indices"},
      {"label", "Label"},
      {"total", "Total"},
      {"x", "Out"},
    }},
    {"acos", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"acosh", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"adadelta_", {
      {"avg_squared_grad", "AvgSquaredGrad"},
      {"avg_squared_update", "AvgSquaredUpdate"},
      {"grad", "Grad"},
      {"inf_norm_out", "AvgSquaredUpdateOut"},
      {"learning_rate", "LearningRate"},
      {"master_param", "MasterParam"},
      {"master_param_out", "MasterParamOut"},
      {"moment_out", "AvgSquaredGradOut"},
      {"param", "Param"},
      {"param_out", "ParamOut"},
    }},
    {"adagrad_", {
      {"grad", "Grad"},
      {"learning_rate", "LearningRate"},
      {"master_param", "MasterParam"},
      {"master_param_out", "MasterParamOut"},
      {"moment", "Moment"},
      {"moment_out", "MomentOut"},
      {"param", "Param"},
      {"param_out", "ParamOut"},
    }},
    {"adam_", {
      {"beta1_pow", "Beta1Pow"},
      {"beta1_pow_out", "Beta1PowOut"},
      {"beta2_pow", "Beta2Pow"},
      {"beta2_pow_out", "Beta2PowOut"},
      {"grad", "Grad"},
      {"learning_rate", "LearningRate"},
      {"master_param", "MasterParam"},
      {"master_param_out", "MasterParamOut"},
      {"moment1", "Moment1"},
      {"moment1_out", "Moment1Out"},
      {"moment2", "Moment2"},
      {"moment2_max", "Moment2Max"},
      {"moment2_max_out", "Moment2MaxOut"},
      {"moment2_out", "Moment2Out"},
      {"param", "Param"},
      {"param_out", "ParamOut"},
      {"skip_update", "SkipUpdate"},
    }},
    {"adamax_", {
      {"beta1_pow", "Beta1Pow"},
      {"grad", "Grad"},
      {"inf_norm", "InfNorm"},
      {"inf_norm_out", "InfNormOut"},
      {"learning_rate", "LearningRate"},
      {"master_param", "MasterParam"},
      {"master_param_out", "MasterParamOut"},
      {"moment", "Moment"},
      {"moment_out", "MomentOut"},
      {"param", "Param"},
      {"param_out", "ParamOut"},
    }},
    {"adamw_", {
      {"beta1_pow", "Beta1Pow"},
      {"beta1_pow_out", "Beta1PowOut"},
      {"beta2_pow", "Beta2Pow"},
      {"beta2_pow_out", "Beta2PowOut"},
      {"grad", "Grad"},
      {"learning_rate", "LearningRate"},
      {"master_param", "MasterParam"},
      {"master_param_out", "MasterParamOut"},
      {"moment1", "Moment1"},
      {"moment1_out", "Moment1Out"},
      {"moment2", "Moment2"},
      {"moment2_max", "Moment2Max"},
      {"moment2_max_out", "Moment2MaxOut"},
      {"moment2_out", "Moment2Out"},
      {"param", "Param"},
      {"param_out", "ParamOut"},
      {"skip_update", "SkipUpdate"},
    }},
    {"add", {
      {"out", "Out"},
      {"x", "X"},
      {"y", "Y"},
    }},
    {"add_n", {
      {"inputs", "X"},
      {"out", "Out"},
    }},
    {"add_position_encoding", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"addmm", {
      {"input", "Input"},
      {"out", "Out"},
      {"x", "X"},
      {"y", "Y"},
    }},
    {"affine_channel", {
      {"bias", "Bias"},
      {"out", "Out"},
      {"scale", "Scale"},
      {"x", "X"},
    }},
    {"affine_grid", {
      {"input", "Theta"},
      {"output", "Output"},
    }},
    {"all", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"allclose", {
      {"out", "Out"},
      {"x", "Input"},
      {"y", "Other"},
    }},
    {"amax", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"amin", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"anchor_generator", {
      {"anchors", "Anchors"},
      {"input", "Input"},
      {"variances_out", "Variances"},
    }},
    {"angle", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"any", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"arange", {
      {"end", "End"},
      {"out", "Out"},
      {"start", "Start"},
      {"step", "Step"},
    }},
    {"argmax", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"argmin", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"argsort", {
      {"indices", "Indices"},
      {"out", "Out"},
      {"x", "X"},
    }},
    {"array_to_tensor", {
      {"out", "Out"},
      {"out_index", "OutIndex"},
      {"x", "X"},
    }},
    {"as_complex", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"as_real", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"asin", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"asinh", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"assert", {
      {"cond", "Cond"},
      {"data", "Data"},
    }},
    {"assign", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"assign_pos", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"assign_value", {
      {"out", "Out"},
    }},
    {"atan", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"atan2", {
      {"out", "Out"},
      {"x", "X1"},
      {"y", "X2"},
    }},
    {"atanh", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"attention_lstm", {
      {"attention_bias", "AttentionBias"},
      {"attention_fc_out", "AttentionFCOut"},
      {"attention_scalar", "AttentionScalar"},
      {"attention_scalar_bias", "AttentionScalarBias"},
      {"attention_weight", "AttentionWeight"},
      {"attentioned_x", "AttentionedX"},
      {"c0", "C0"},
      {"cell", "Cell"},
      {"h0", "H0"},
      {"hidden", "Hidden"},
      {"lstm_bias", "LSTMBias"},
      {"lstm_out", "LSTMOUT"},
      {"lstm_weight", "LSTMWeight"},
      {"lstm_x", "LSTMX"},
      {"x", "X"},
    }},
    {"auc", {
      {"auc", "AUC"},
      {"ins_tag_weight", "InsTagWeight"},
      {"label", "Label"},
      {"stat_neg", "StatNeg"},
      {"stat_neg_out", "StatNegOut"},
      {"stat_pos", "StatPos"},
      {"stat_pos_out", "StatPosOut"},
      {"x", "Predict"},
    }},
    {"baddbmm", {
      {"input", "Input"},
      {"out", "Out"},
      {"x", "X"},
      {"y", "Y"},
    }},
    {"barrier", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"batch_fc", {
      {"bias", "Bias"},
      {"input", "Input"},
      {"out", "Out"},
      {"w", "W"},
    }},
    {"batch_norm", {
      {"bias", "Bias"},
      {"mean", "Mean"},
      {"mean_out", "MeanOut"},
      {"out", "Y"},
      {"reserve_space", "ReserveSpace"},
      {"saved_mean", "SavedMean"},
      {"saved_variance", "SavedVariance"},
      {"scale", "Scale"},
      {"variance", "Variance"},
      {"variance_out", "VarianceOut"},
      {"x", "X"},
    }},
    {"bce_loss", {
      {"input", "X"},
      {"label", "Label"},
      {"out", "Out"},
    }},
    {"beam_search_decode", {
      {"ids", "Ids"},
      {"scores", "Scores"},
      {"sentence_ids", "SentenceIds"},
      {"sentence_scores", "SentenceScores"},
    }},
    {"bernoulli", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"bicubic_interp", {
      {"out_size", "OutSize"},
      {"output", "Out"},
      {"scale_tensor", "Scale"},
      {"size_tensor", "SizeTensor"},
      {"x", "X"},
    }},
    {"bilinear", {
      {"bias", "Bias"},
      {"out", "Out"},
      {"weight", "Weight"},
      {"x", "X"},
      {"y", "Y"},
    }},
    {"bilinear_interp", {
      {"out_size", "OutSize"},
      {"output", "Out"},
      {"scale_tensor", "Scale"},
      {"size_tensor", "SizeTensor"},
      {"x", "X"},
    }},
    {"bincount", {
      {"out", "Out"},
      {"weights", "Weights"},
      {"x", "X"},
    }},
    {"bipartite_match", {
      {"col_to_row_match_dist", "ColToRowMatchDist"},
      {"col_to_row_match_indices", "ColToRowMatchIndices"},
      {"dist_mat", "DistMat"},
    }},
    {"bitwise_and", {
      {"out", "Out"},
      {"x", "X"},
      {"y", "Y"},
    }},
    {"bitwise_not", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"bitwise_or", {
      {"out", "Out"},
      {"x", "X"},
      {"y", "Y"},
    }},
    {"bitwise_xor", {
      {"out", "Out"},
      {"x", "X"},
      {"y", "Y"},
    }},
    {"bmm", {
      {"out", "Out"},
      {"x", "X"},
      {"y", "Y"},
    }},
    {"box_clip", {
      {"im_info", "ImInfo"},
      {"input", "Input"},
      {"output", "Output"},
    }},
    {"box_coder", {
      {"output_box", "OutputBox"},
      {"prior_box", "PriorBox"},
      {"prior_box_var", "PriorBoxVar"},
      {"target_box", "TargetBox"},
    }},
    {"broadcast_tensors", {
      {"input", "X"},
      {"out", "Out"},
    }},
    {"c_allreduce_sum", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"c_concat", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"c_embedding", {
      {"out", "Out"},
      {"weight", "W"},
      {"x", "Ids"},
    }},
    {"c_identity", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"c_scatter", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"c_softmax_with_cross_entropy", {
      {"label", "Label"},
      {"logits", "Logits"},
      {"loss", "Loss"},
      {"softmax", "Softmax"},
    }},
    {"c_split", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"cast", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"ceil", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"celu", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"channel_shuffle", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"check_finite_and_unscale_", {
      {"found_infinite", "FoundInfinite"},
      {"out", "Out"},
      {"scale", "Scale"},
      {"x", "X"},
    }},
    {"cholesky", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"cholesky_solve", {
      {"out", "Out"},
      {"x", "X"},
      {"y", "Y"},
    }},
    {"chunk_eval", {
      {"f1_score", "F1-Score"},
      {"inference", "Inference"},
      {"label", "Label"},
      {"num_correct_chunks", "NumCorrectChunks"},
      {"num_infer_chunks", "NumInferChunks"},
      {"num_label_chunks", "NumLabelChunks"},
      {"precision", "Precision"},
      {"recall", "Recall"},
      {"seq_length", "SeqLength"},
    }},
    {"class_center_sample", {
      {"label", "Label"},
      {"remapped_label", "RemappedLabel"},
      {"sampled_local_class_center", "SampledLocalClassCenter"},
    }},
    {"clip", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"clip_by_norm", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"coalesce_tensor", {
      {"fused_output", "FusedOutput"},
      {"input", "Input"},
      {"output", "Output"},
    }},
    {"collect_fpn_proposals", {
      {"fpn_rois", "FpnRois"},
      {"multi_level_rois", "MultiLevelRois"},
      {"multi_level_rois_num", "MultiLevelRoIsNum"},
      {"multi_level_scores", "MultiLevelScores"},
      {"rois_num", "RoisNum"},
    }},
    {"complex", {
      {"imag", "Y"},
      {"out", "Out"},
      {"real", "X"},
    }},
    {"concat", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"conj", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"conv2d", {
      {"filter", "Filter"},
      {"input", "Input"},
      {"out", "Output"},
    }},
    {"conv2d_transpose", {
      {"bias", "Bias"},
      {"filter", "Filter"},
      {"out", "Output"},
      {"x", "Input"},
    }},
    {"conv2d_transpose_bias", {
      {"bias", "Bias"},
      {"filter", "Filter"},
      {"out", "Output"},
      {"x", "Input"},
    }},
    {"conv3d", {
      {"filter", "Filter"},
      {"input", "Input"},
      {"out", "Output"},
    }},
    {"conv3d_transpose", {
      {"filter", "Filter"},
      {"out", "Output"},
      {"x", "Input"},
    }},
    {"correlation", {
      {"input1", "Input1"},
      {"input2", "Input2"},
      {"out", "Output"},
    }},
    {"cos", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"cosh", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"crf_decoding", {
      {"emission", "Emission"},
      {"label", "Label"},
      {"length", "Length"},
      {"transition", "Transition"},
      {"viterbi_path", "ViterbiPath"},
    }},
    {"crop", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"cross", {
      {"out", "Out"},
      {"x", "X"},
      {"y", "Y"},
    }},
    {"cross_entropy", {
      {"label", "Label"},
      {"out", "Y"},
      {"x", "X"},
    }},
    {"cross_entropy2", {
      {"label", "Label"},
      {"match_x", "MatchX"},
      {"out", "Y"},
      {"x", "X"},
      {"x_shape", "XShape"},
    }},
    {"cross_entropy_with_softmax", {
      {"input", "Logits"},
      {"label", "Label"},
      {"loss", "Loss"},
      {"softmax", "Softmax"},
    }},
    {"ctc_align", {
      {"input", "Input"},
      {"input_length", "InputLength"},
      {"output", "Output"},
      {"output_length", "OutputLength"},
    }},
    {"cudnn_lstm", {
      {"init_c", "InitC"},
      {"init_h", "InitH"},
      {"last_c", "LastC"},
      {"last_h", "LastH"},
      {"out", "Out"},
      {"reserve", "Reserve"},
      {"sequence_length", "SequenceLength"},
      {"state_out", "StateOut"},
      {"w", "W"},
      {"weight_list", "WeightList"},
      {"x", "Input"},
    }},
    {"cumprod", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"cumsum", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"cvm", {
      {"cvm", "CVM"},
      {"out", "Y"},
      {"x", "X"},
    }},
    {"decayed_adagrad", {
      {"grad", "Grad"},
      {"learning_rate", "LearningRate"},
      {"moment", "Moment"},
      {"moment_out", "MomentOut"},
      {"param", "Param"},
      {"param_out", "ParamOut"},
    }},
    {"decode_jpeg", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"deformable_conv", {
      {"filter", "Filter"},
      {"mask", "Mask"},
      {"offset", "Offset"},
      {"out", "Output"},
      {"x", "Input"},
    }},
    {"depend", {
      {"dep", "Dep"},
      {"out", "Out"},
      {"x", "X"},
    }},
    {"depthwise_conv2d", {
      {"filter", "Filter"},
      {"input", "Input"},
      {"out", "Output"},
    }},
    {"depthwise_conv2d_transpose", {
      {"bias", "Bias"},
      {"filter", "Filter"},
      {"out", "Output"},
      {"x", "Input"},
    }},
    {"dequantize", {
      {"input", "Input"},
      {"output", "Output"},
    }},
    {"dequantize_abs_max", {
      {"out", "Out"},
      {"scale", "Scale"},
      {"x", "X"},
    }},
    {"dequantize_linear", {
      {"in_accum", "InAccum"},
      {"in_state", "InState"},
      {"out_accum", "OutAccum"},
      {"out_scale", "OutScale"},
      {"out_state", "OutState"},
      {"scale", "Scale"},
      {"x", "X"},
      {"y", "Y"},
      {"zero_point", "ZeroPoint"},
    }},
    {"dequantize_log", {
      {"dict", "Dict"},
      {"out", "Out"},
      {"x", "X"},
    }},
    {"det", {
      {"out", "Out"},
      {"x", "Input"},
    }},
    {"dgc", {
      {"encode_grad", "EncodeGrad"},
      {"gather_buff", "GatherBuff"},
      {"grad", "Grad"},
      {"grad_out", "Grad_out"},
      {"param", "Param"},
      {"u", "U"},
      {"u_out", "U_out"},
      {"v", "V"},
      {"v_out", "V_out"},
    }},
    {"dgc_clip_by_norm", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"dgc_momentum", {
      {"current_step_tensor", "current_step"},
      {"grad", "Grad"},
      {"grad_out", "Grad_out"},
      {"learning_rate", "LearningRate"},
      {"master_param", "MasterParam"},
      {"master_param_out", "MasterParamOut"},
      {"nranks_tensor", "nranks"},
      {"param", "Param"},
      {"param_out", "ParamOut"},
      {"velocity", "Velocity"},
      {"velocity_out", "VelocityOut"},
    }},
    {"diag", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"diag_embed", {
      {"input", "Input"},
      {"out", "Out"},
    }},
    {"diagonal", {
      {"out", "Out"},
      {"x", "Input"},
    }},
    {"digamma", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"dirichlet", {
      {"alpha", "Alpha"},
      {"out", "Out"},
    }},
    {"dist", {
      {"out", "Out"},
      {"x", "X"},
      {"y", "Y"},
    }},
    {"distribute_fpn_proposals", {
      {"fpn_rois", "FpnRois"},
      {"multi_fpn_rois", "MultiFpnRois"},
      {"multi_level_rois_num", "MultiLevelRoIsNum"},
      {"restore_index", "RestoreIndex"},
      {"rois_num", "RoisNum"},
    }},
    {"distributed_fused_lamb_init", {
      {"beta1_pow", "Beta1Pow"},
      {"beta2_pow", "Beta2Pow"},
      {"fp16_fused_grad", "FP16FusedGrad"},
      {"fp16_fused_param", "FP16FusedParam"},
      {"fp16_shard_fused_param_offsets", "FP16ShardFusedParamOffsets"},
      {"fp32_fused_grad", "FP32FusedGrad"},
      {"fp32_fused_param", "FP32FusedParam"},
      {"fp32_shard_fused_param_offsets", "FP32ShardFusedParamOffsets"},
      {"fused_param_offsets", "FusedParamOffsets"},
      {"global_scale", "GlobalScale"},
      {"grad", "Grad"},
      {"grad_out", "GradOut"},
      {"master_param_out", "MasterParamOut"},
      {"moment1", "Moment1"},
      {"moment2", "Moment2"},
      {"param", "Param"},
      {"param_info", "ParamInfo"},
      {"param_order", "ParamOrder"},
      {"param_out", "ParamOut"},
      {"step", "Step"},
    }},
    {"div_scale", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"divide", {
      {"out", "Out"},
      {"x", "X"},
      {"y", "Y"},
    }},
    {"dot", {
      {"out", "Out"},
      {"x", "X"},
      {"y", "Y"},
    }},
    {"dpsgd", {
      {"grad", "Grad"},
      {"learning_rate", "LearningRate"},
      {"param", "Param"},
      {"param_out", "ParamOut"},
    }},
    {"dropout", {
      {"mask", "Mask"},
      {"out", "Out"},
      {"seed_tensor", "Seed"},
      {"x", "X"},
    }},
    {"edit_distance", {
      {"hyps", "Hyps"},
      {"hypslength", "HypsLength"},
      {"out", "Out"},
      {"refs", "Refs"},
      {"refslength", "RefsLength"},
      {"sequencenum", "SequenceNum"},
    }},
    {"eig", {
      {"out_v", "Eigenvectors"},
      {"out_w", "Eigenvalues"},
      {"x", "X"},
    }},
    {"eigh", {
      {"out_v", "Eigenvectors"},
      {"out_w", "Eigenvalues"},
      {"x", "X"},
    }},
    {"eigvals", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"eigvalsh", {
      {"eigenvalues", "Eigenvalues"},
      {"eigenvectors", "Eigenvectors"},
      {"x", "X"},
    }},
    {"einsum", {
      {"inner_cache", "InnerCache"},
      {"out", "Out"},
      {"x", "Operands"},
      {"xshape", "XShape"},
    }},
    {"elementwise_pow", {
      {"out", "Out"},
      {"x", "X"},
      {"y", "Y"},
    }},
    {"elu", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"embedding", {
      {"out", "Out"},
      {"weight", "W"},
      {"x", "Ids"},
    }},
    {"empty", {
      {"out", "Out"},
    }},
    {"equal", {
      {"out", "Out"},
      {"x", "X"},
      {"y", "Y"},
    }},
    {"equal_all", {
      {"out", "Out"},
      {"x", "X"},
      {"y", "Y"},
    }},
    {"erf", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"erfinv", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"exp", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"expand", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"expand_as", {
      {"out", "Out"},
      {"x", "X"},
      {"y", "Y"},
    }},
    {"expm1", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"exponential_", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"eye", {
      {"out", "Out"},
    }},
    {"fake_channel_wise_dequantize_max_abs", {
      {"out", "Out"},
      {"scales", "Scales"},
      {"x", "X"},
    }},
    {"fake_channel_wise_quantize_abs_max", {
      {"out", "Out"},
      {"out_scale", "OutScale"},
      {"x", "X"},
    }},
    {"fake_channel_wise_quantize_dequantize_abs_max", {
      {"out", "Out"},
      {"out_scale", "OutScale"},
      {"x", "X"},
    }},
    {"fake_dequantize_max_abs", {
      {"out", "Out"},
      {"scale", "Scale"},
      {"x", "X"},
    }},
    {"fake_quantize_abs_max", {
      {"out", "Out"},
      {"out_scale", "OutScale"},
      {"x", "X"},
    }},
    {"fake_quantize_dequantize_abs_max", {
      {"out", "Out"},
      {"out_scale", "OutScale"},
      {"x", "X"},
    }},
    {"fake_quantize_dequantize_moving_average_abs_max", {
      {"in_accum", "InAccum"},
      {"in_scale", "InScale"},
      {"in_state", "InState"},
      {"out", "Out"},
      {"out_accum", "OutAccum"},
      {"out_scale", "OutScale"},
      {"out_state", "OutState"},
      {"x", "X"},
    }},
    {"fake_quantize_moving_average_abs_max", {
      {"in_accum", "InAccum"},
      {"in_scale", "InScale"},
      {"in_state", "InState"},
      {"out", "Out"},
      {"out_accum", "OutAccum"},
      {"out_scale", "OutScale"},
      {"out_state", "OutState"},
      {"x", "X"},
    }},
    {"fake_quantize_range_abs_max", {
      {"in_scale", "InScale"},
      {"iter", "Iter"},
      {"out", "Out"},
      {"out_scale", "OutScale"},
      {"out_scales", "OutScales"},
      {"x", "X"},
    }},
    {"faster_tokenizer", {
      {"input_ids", "InputIds"},
      {"segment_ids", "SegmentIds"},
      {"text", "Text"},
      {"text_pair", "TextPair"},
      {"vocab", "Vocab"},
    }},
    {"fc", {
      {"bias", "Bias"},
      {"input", "Input"},
      {"out", "Out"},
      {"w", "W"},
    }},
    {"feed", {
      {"out", "Out"},
    }},
    {"fetch", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"fetch_barrier", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"fft_c2c", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"fft_c2r", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"fft_r2c", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"fill", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"fill_diagonal", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"fill_diagonal_tensor", {
      {"out", "Out"},
      {"x", "X"},
      {"y", "Y"},
    }},
    {"flatten", {
      {"out", "Out"},
      {"x", "X"},
      {"xshape", "XShape"},
    }},
    {"flatten2", {
      {"out", "Out"},
      {"x", "X"},
      {"x_shape", "XShape"},
    }},
    {"flip", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"floor", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"floor_divide", {
      {"out", "Out"},
      {"x", "X"},
      {"y", "Y"},
    }},
    {"fmax", {
      {"out", "Out"},
      {"x", "X"},
      {"y", "Y"},
    }},
    {"fmin", {
      {"out", "Out"},
      {"x", "X"},
      {"y", "Y"},
    }},
    {"fold", {
      {"out", "Y"},
      {"x", "X"},
    }},
    {"frame", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"frobenius_norm", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"ftrl", {
      {"grad", "Grad"},
      {"learning_rate", "LearningRate"},
      {"linear_accum_out", "LinearAccumOut"},
      {"linear_accumulator", "LinearAccumulator"},
      {"param", "Param"},
      {"param_out", "ParamOut"},
      {"squared_accum_out", "SquaredAccumOut"},
      {"squared_accumulator", "SquaredAccumulator"},
    }},
    {"full", {
      {"out", "Out"},
    }},
    {"full_batch_size_like", {
      {"input", "Input"},
      {"out", "Out"},
    }},
    {"full_like", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"fused_adam_", {
      {"beta1_pows", "Beta1Pows"},
      {"beta1_pows_out", "Beta1PowsOut"},
      {"beta2_pows", "Beta2Pows"},
      {"beta2_pows_out", "Beta2PowsOut"},
      {"grads", "Grads"},
      {"learning_rate", "LearningRate"},
      {"master_params", "MasterParams"},
      {"master_params_out", "MasterParamsOut"},
      {"moments1", "Moments1"},
      {"moments1_out", "Moments1Out"},
      {"moments2", "Moments2"},
      {"moments2_max", "Moments2Max"},
      {"moments2_max_out", "Moments2MaxOut"},
      {"moments2_out", "Moments2Out"},
      {"params", "Params"},
      {"params_out", "ParamsOut"},
      {"skip_update", "SkipUpdate"},
    }},
    {"fused_attention", {
      {"attn_dropout_mask_out", "AttnDropoutMaskOut"},
      {"attn_dropout_out", "AttnDropoutOut"},
      {"bias_dropout_residual_out", "BiasDropoutResidualOut"},
      {"cache_kv", "CacheKV"},
      {"cache_kv_out", "CacheKVOut"},
      {"dropout_mask_out", "DropoutMaskOut"},
      {"fmha_out", "FMHAOut"},
      {"ln_bias", "LnBias"},
      {"ln_bias_2", "Ln2Bias"},
      {"ln_mean", "LnMean"},
      {"ln_mean_2", "Ln2Mean"},
      {"ln_out", "LnOut"},
      {"ln_scale", "LnScale"},
      {"ln_scale_2", "Ln2Scale"},
      {"ln_var", "LnVariance"},
      {"ln_var_2", "Ln2Variance"},
      {"out", "Y"},
      {"out_linear_bias", "OutLinearBias"},
      {"out_linear_out", "OutLinearOut"},
      {"out_linear_weight", "OutLinearW"},
      {"qk_out", "QKOut"},
      {"qktv_out", "QKTVOut"},
      {"qkv_bias", "QKVBias"},
      {"qkv_bias_out", "QKVBiasOut"},
      {"qkv_out", "QKVOut"},
      {"qkv_weight", "QKVW"},
      {"softmax_out", "SoftmaxOut"},
      {"src_mask", "SrcMask"},
      {"src_mask_out", "SrcMaskOut"},
      {"transpose_out_2", "TransposeOut2"},
      {"x", "X"},
    }},
    {"fused_batch_norm_act", {
      {"bias", "Bias"},
      {"mean", "Mean"},
      {"mean_out", "MeanOut"},
      {"out", "Y"},
      {"reserve_space", "ReserveSpace"},
      {"saved_mean", "SavedMean"},
      {"saved_variance", "SavedVariance"},
      {"scale", "Scale"},
      {"variance", "Variance"},
      {"variance_out", "VarianceOut"},
      {"x", "X"},
    }},
    {"fused_bias_dropout_residual_layer_norm", {
      {"bias", "Bias"},
      {"bias_dropout_residual_out", "BiasDropoutResidualOut"},
      {"dropout_mask_out", "DropoutMaskOut"},
      {"ln_bias", "LnBias"},
      {"ln_mean", "LnMean"},
      {"ln_scale", "LnScale"},
      {"ln_variance", "LnVariance"},
      {"residual", "Residual"},
      {"x", "X"},
      {"y", "Y"},
    }},
    {"fused_bn_add_activation_", {
      {"bias", "Bias"},
      {"mean", "Mean"},
      {"mean_out", "MeanOut"},
      {"out", "Y"},
      {"reserve_space", "ReserveSpace"},
      {"saved_mean", "SavedMean"},
      {"saved_variance", "SavedVariance"},
      {"scale", "Scale"},
      {"variance", "Variance"},
      {"variance_out", "VarianceOut"},
      {"x", "X"},
      {"z", "Z"},
    }},
    {"fused_conv2d", {
      {"bias", "Bias"},
      {"filter", "Filter"},
      {"input", "Input"},
      {"output", "Output"},
      {"residual_param", "ResidualData"},
    }},
    {"fused_conv2d_add_act", {
      {"bias", "Bias"},
      {"filter", "Filter"},
      {"input", "Input"},
      {"output", "Output"},
      {"outputs", "Outputs"},
      {"residual_data", "ResidualData"},
    }},
    {"fused_conv3d", {
      {"bias", "Bias"},
      {"filter", "Filter"},
      {"input", "Input"},
      {"output", "Output"},
      {"residual_param", "ResidualData"},
    }},
    {"fused_elementwise_add", {
      {"out", "Out"},
      {"x", "X"},
      {"y", "Y"},
    }},
    {"fused_elementwise_div", {
      {"out", "Out"},
      {"x", "X"},
      {"y", "Y"},
    }},
    {"fused_elementwise_mul", {
      {"out", "Out"},
      {"x", "X"},
      {"y", "Y"},
    }},
    {"fused_elementwise_sub", {
      {"out", "Out"},
      {"x", "X"},
      {"y", "Y"},
    }},
    {"fused_elemwise_activation", {
      {"intermediate_out", "IntermediateOut"},
      {"out", "Out"},
      {"x", "X"},
      {"y", "Y"},
    }},
    {"fused_elemwise_add_activation", {
      {"intermediate_out", "IntermediateOut"},
      {"out", "Out"},
      {"x", "X"},
      {"y", "Y"},
    }},
    {"fused_embedding_eltwise_layernorm", {
      {"bias", "Bias"},
      {"embs", "Embs"},
      {"ids", "Ids"},
      {"out", "Out"},
      {"scale", "Scale"},
    }},
    {"fused_embedding_fc_lstm", {
      {"batched_cell", "BatchedCell"},
      {"batched_hidden", "BatchedHidden"},
      {"batched_input", "BatchedInput"},
      {"bias", "Bias"},
      {"c0", "C0"},
      {"cell", "Cell"},
      {"embeddings", "Embeddings"},
      {"h0", "H0"},
      {"hidden", "Hidden"},
      {"ids", "Ids"},
      {"reordered_c0", "ReorderedC0"},
      {"reordered_h0", "ReorderedH0"},
      {"weight_h", "WeightH"},
      {"xx", "XX"},
    }},
    {"fused_fc_elementwise_layernorm", {
      {"bias0", "Bias0"},
      {"bias1", "Bias1"},
      {"mean", "Mean"},
      {"out", "Out"},
      {"scale", "Scale"},
      {"variance", "Variance"},
      {"w", "W"},
      {"x", "X"},
      {"y", "Y"},
    }},
    {"fused_feedforward", {
      {"dropout1_mask", "Dropout1Mask"},
      {"dropout1_out", "Dropout1Out"},
      {"dropout1_seed", "Dropout1Seed"},
      {"dropout2_mask", "Dropout2Mask"},
      {"dropout2_out", "Dropout2Out"},
      {"dropout2_seed", "Dropout2Seed"},
      {"linear1_bias", "Linear1Bias"},
      {"linear1_out", "Linear1Out"},
      {"linear1_weight", "Linear1Weight"},
      {"linear2_bias", "Linear2Bias"},
      {"linear2_weight", "Linear2Weight"},
      {"ln1_bias", "Ln1Bias"},
      {"ln1_mean", "Ln1Mean"},
      {"ln1_out", "Ln1Out"},
      {"ln1_scale", "Ln1Scale"},
      {"ln1_variance", "Ln1Variance"},
      {"ln2_bias", "Ln2Bias"},
      {"ln2_mean", "Ln2Mean"},
      {"ln2_scale", "Ln2Scale"},
      {"ln2_variance", "Ln2Variance"},
      {"out", "Out"},
      {"x", "X"},
    }},
    {"fused_gate_attention", {
      {"fmha_out", "FMHAOut"},
      {"gate_bias", "GateBias"},
      {"gate_out", "GateOut"},
      {"gate_weight", "GateWeight"},
      {"key", "Key"},
      {"key_transpose_out", "KeyTransposeOut"},
      {"key_weight", "KeyWeight"},
      {"nonbatched_bias", "NonbatchedBias"},
      {"out", "Out"},
      {"out_linear_bias", "OutLinearBias"},
      {"out_linear_weight", "OutLinearWeight"},
      {"qkv_transpose_out", "QKVTransposeOut"},
      {"qkv_weight", "QKVWeight"},
      {"query", "Query"},
      {"query_transpose_out", "QueryTransposeOut"},
      {"query_weight", "QueryWeight"},
      {"softmax_lse", "SoftmaxLse"},
      {"softmax_out", "SoftmaxOut"},
      {"src_mask", "SrcMask"},
      {"value_transpose_out", "ValueTransposeOut"},
      {"value_weight", "ValueWeight"},
    }},
    {"fused_gemm_epilogue", {
      {"bias", "Bias"},
      {"out", "Out"},
      {"reserve_space", "ReserveSpace"},
      {"x", "X"},
      {"y", "Y"},
    }},
    {"fused_gemm_epilogue_grad", {
      {"bias_grad", "DBias"},
      {"out_grad", "DOut"},
      {"reserve_space", "ReserveSpace"},
      {"x", "X"},
      {"x_grad", "DX"},
      {"y", "Y"},
      {"y_grad", "DY"},
    }},
    {"fused_matmul", {
      {"out", "Out"},
      {"residual_data", "ResidualData"},
      {"x", "X"},
      {"y", "Y"},
    }},
    {"fused_multi_transformer_int8", {
      {"cache_kv", "CacheKV"},
      {"cache_kv_out", "CacheKVOut"},
      {"ffn1_bias", "FFN1Bias"},
      {"ffn1_out_scale", "FFN1OutScale"},
      {"ffn1_weight", "FFN1Weight"},
      {"ffn2_bias", "FFN2Bias"},
      {"ffn2_out_scale", "FFN2OutScale"},
      {"ffn2_weight", "FFN2Weight"},
      {"ffn_ln_bias", "FFNLnBias"},
      {"ffn_ln_scale", "FFNLnScale"},
      {"ln_bias", "LnBias"},
      {"ln_scale", "LnScale"},
      {"out", "Out"},
      {"out_linear_bias", "OutLinearBias"},
      {"out_linear_out_scale", "OutLinearOutScale"},
      {"out_linear_w", "OutLinearW"},
      {"qkv_bias", "QKVBias"},
      {"qkv_out_scale", "QKVOutScale"},
      {"qkv_w", "QKVW"},
      {"src_mask", "SrcMask"},
      {"time_step", "TimeStep"},
      {"x", "X"},
    }},
    {"fused_seqpool_cvm", {
      {"cvm", "CVM"},
      {"out", "Out"},
      {"x", "X"},
    }},
    {"fused_softmax_mask", {
      {"mask", "Mask"},
      {"out", "Out"},
      {"x", "X"},
    }},
    {"fused_softplus", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"fused_token_prune", {
      {"attn", "Attn"},
      {"cls_inds", "CLSInds"},
      {"mask", "Mask"},
      {"new_mask", "NewMask"},
      {"slimmed_x", "SlimmedX"},
      {"x", "X"},
    }},
    {"fused_transpose", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"fusion_group", {
      {"inputs", "Inputs"},
      {"outs", "Outs"},
    }},
    {"fusion_gru", {
      {"batched_input", "BatchedInput"},
      {"batched_out", "BatchedOut"},
      {"bias", "Bias"},
      {"h0", "H0"},
      {"hidden", "Hidden"},
      {"reordered_h0", "ReorderedH0"},
      {"weight_h", "WeightH"},
      {"weight_x", "WeightX"},
      {"x", "X"},
      {"xx", "XX"},
    }},
    {"fusion_lstm", {
      {"batched_cell", "BatchedCell"},
      {"batched_hidden", "BatchedHidden"},
      {"batched_input", "BatchedInput"},
      {"bias", "Bias"},
      {"c0", "C0"},
      {"cell", "Cell"},
      {"checked_cell", "CheckedCell"},
      {"h0", "H0"},
      {"hidden", "Hidden"},
      {"out", "Out"},
      {"reordered_c0", "ReorderedC0"},
      {"reordered_h0", "ReorderedH0"},
      {"weight_h", "WeightH"},
      {"weight_x", "WeightX"},
      {"x", "X"},
      {"xx", "XX"},
    }},
    {"fusion_repeated_fc_relu", {
      {"bias", "Bias"},
      {"out", "Out"},
      {"relu_out", "ReluOut"},
      {"w", "W"},
      {"x", "X"},
    }},
    {"fusion_seqconv_eltadd_relu", {
      {"bias", "Bias"},
      {"col_mat", "ColMat"},
      {"filter", "Filter"},
      {"out", "Out"},
      {"x", "X"},
    }},
    {"fusion_seqpool_concat", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"fusion_seqpool_cvm_concat", {
      {"cvm", "CVM"},
      {"out", "Out"},
      {"x", "X"},
    }},
    {"fusion_squared_mat_sub", {
      {"out", "Out"},
      {"squared_x", "SquaredX"},
      {"squared_xy", "SquaredXY"},
      {"squared_y", "SquaredY"},
      {"x", "X"},
      {"y", "Y"},
    }},
    {"fusion_transpose_flatten_concat", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"gather", {
      {"index", "Index"},
      {"out", "Out"},
      {"x", "X"},
    }},
    {"gather_nd", {
      {"index", "Index"},
      {"out", "Out"},
      {"x", "X"},
    }},
    {"gather_tree", {
      {"ids", "Ids"},
      {"out", "Out"},
      {"parents", "Parents"},
    }},
    {"gaussian", {
      {"out", "Out"},
    }},
    {"gelu", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"generate_proposals", {
      {"anchors", "Anchors"},
      {"bbox_deltas", "BboxDeltas"},
      {"im_shape", "ImShape"},
      {"rpn_roi_probs", "RpnRoiProbs"},
      {"rpn_rois", "RpnRois"},
      {"rpn_rois_num", "RpnRoisNum"},
      {"scores", "Scores"},
      {"variances", "Variances"},
    }},
    {"get_tensor_from_selected_rows", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"global_gather", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"global_scatter", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"grad_add", {
      {"out", "Out"},
      {"x", "X"},
      {"y", "Y"},
    }},
    {"graph_khop_sampler", {
      {"colptr", "Col_Ptr"},
      {"eids", "Eids"},
      {"out_dst", "Out_Dst"},
      {"out_eids", "Out_Eids"},
      {"out_src", "Out_Src"},
      {"reindex_x", "Reindex_X"},
      {"row", "Row"},
      {"sample_index", "Sample_Index"},
      {"x", "X"},
    }},
    {"graph_sample_neighbors", {
      {"colptr", "Col_Ptr"},
      {"eids", "Eids"},
      {"out", "Out"},
      {"out_count", "Out_Count"},
      {"out_eids", "Out_Eids"},
      {"perm_buffer", "Perm_Buffer"},
      {"row", "Row"},
      {"x", "X"},
    }},
    {"greater_equal", {
      {"out", "Out"},
      {"x", "X"},
      {"y", "Y"},
    }},
    {"greater_than", {
      {"out", "Out"},
      {"x", "X"},
      {"y", "Y"},
    }},
    {"grid_sample", {
      {"grid", "Grid"},
      {"out", "Output"},
      {"x", "X"},
    }},
    {"group_norm", {
      {"bias", "Bias"},
      {"mean", "Mean"},
      {"scale", "Scale"},
      {"variance", "Variance"},
      {"x", "X"},
      {"y", "Y"},
    }},
    {"gru", {
      {"batch_gate", "BatchGate"},
      {"batch_hidden", "BatchHidden"},
      {"batch_reset_hidden_prev", "BatchResetHiddenPrev"},
      {"bias", "Bias"},
      {"h0", "H0"},
      {"hidden", "Hidden"},
      {"input", "Input"},
      {"weight", "Weight"},
    }},
    {"gru_unit", {
      {"bias", "Bias"},
      {"gate", "Gate"},
      {"hidden", "Hidden"},
      {"hidden_prev", "HiddenPrev"},
      {"input", "Input"},
      {"reset_hidden_prev", "ResetHiddenPrev"},
      {"weight", "Weight"},
    }},
    {"gumbel_softmax", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"hardshrink", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"hardsigmoid", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"hardswish", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"hardtanh", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"hash", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"heaviside", {
      {"out", "Out"},
      {"x", "X"},
      {"y", "Y"},
    }},
    {"hinge_loss", {
      {"labels", "Labels"},
      {"logits", "Logits"},
      {"loss", "Loss"},
    }},
    {"histogram", {
      {"input", "X"},
      {"out", "Out"},
      {"weight", "Weight"},
    }},
    {"hsigmoid_loss", {
      {"bias", "Bias"},
      {"code", "PathCode"},
      {"label", "Label"},
      {"out", "Out"},
      {"path", "PathTable"},
      {"pre_out", "PreOut"},
      {"w", "W"},
      {"w_out", "W_Out"},
      {"x", "X"},
    }},
    {"huber_loss", {
      {"input", "X"},
      {"label", "Y"},
      {"out", "Out"},
      {"residual", "Residual"},
    }},
    {"identity_loss", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"im2sequence", {
      {"out", "Out"},
      {"x", "X"},
      {"y", "Y"},
    }},
    {"imag", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"increment", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"index_add", {
      {"add_value", "AddValue"},
      {"index", "Index"},
      {"out", "Out"},
      {"x", "X"},
    }},
    {"index_elementwise_get", {
      {"index", "index"},
      {"index_dims", "index_dims"},
      {"index_stride", "index_stride"},
      {"input_dims", "input_dims"},
      {"input_strides", "input_strides"},
      {"out", "Out"},
      {"x", "x"},
    }},
    {"index_elementwise_put_with_tensor", {
      {"out", "Out"},
    }},
    {"index_sample", {
      {"index", "Index"},
      {"out", "Out"},
      {"x", "X"},
    }},
    {"index_select", {
      {"index", "Index"},
      {"out", "Out"},
      {"x", "X"},
    }},
    {"instance_norm", {
      {"bias", "Bias"},
      {"saved_mean", "SavedMean"},
      {"saved_variance", "SavedVariance"},
      {"scale", "Scale"},
      {"x", "X"},
      {"y", "Y"},
    }},
    {"inverse", {
      {"out", "Output"},
      {"x", "Input"},
    }},
    {"is_empty", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"isclose", {
      {"out", "Out"},
      {"x", "Input"},
      {"y", "Other"},
    }},
    {"isfinite", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"isinf", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"isnan", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"kldiv_loss", {
      {"label", "Target"},
      {"out", "Loss"},
      {"x", "X"},
    }},
    {"kron", {
      {"out", "Out"},
      {"x", "X"},
      {"y", "Y"},
    }},
    {"kthvalue", {
      {"indices", "Indices"},
      {"out", "Out"},
      {"x", "X"},
    }},
    {"l1_norm", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"label_smooth", {
      {"label", "X"},
      {"out", "Out"},
      {"prior_dist", "PriorDist"},
    }},
    {"lamb_", {
      {"beta1_pow", "Beta1Pow"},
      {"beta1_pow_out", "Beta1PowOut"},
      {"beta2_pow", "Beta2Pow"},
      {"beta2_pow_out", "Beta2PowOut"},
      {"grad", "Grad"},
      {"learning_rate", "LearningRate"},
      {"master_param", "MasterParam"},
      {"master_param_outs", "MasterParamOut"},
      {"moment1", "Moment1"},
      {"moment1_out", "Moment1Out"},
      {"moment2", "Moment2"},
      {"moment2_out", "Moment2Out"},
      {"param", "Param"},
      {"param_out", "ParamOut"},
      {"skip_update", "SkipUpdate"},
    }},
    {"lars_momentum_", {
      {"grad", "Grad"},
      {"learning_rate", "LearningRate"},
      {"master_param", "MasterParam"},
      {"master_param_out", "MasterParamOut"},
      {"param", "Param"},
      {"param_out", "ParamOut"},
      {"velocity", "Velocity"},
      {"velocity_out", "VelocityOut"},
    }},
    {"layer_norm", {
      {"bias", "Bias"},
      {"mean", "Mean"},
      {"out", "Y"},
      {"scale", "Scale"},
      {"variance", "Variance"},
      {"x", "X"},
    }},
    {"leaky_relu", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"legacy_bilinear_interp", {
      {"out_size", "OutSize"},
      {"output", "Out"},
      {"scale_tensor", "Scale"},
      {"size_tensor", "SizeTensor"},
      {"x", "X"},
    }},
    {"legacy_crop", {
      {"out", "Out"},
      {"x", "X"},
      {"y", "Y"},
    }},
    {"legacy_expand", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"legacy_generate_proposals", {
      {"anchors", "Anchors"},
      {"bbox_deltas", "BboxDeltas"},
      {"im_info", "ImInfo"},
      {"rpn_roi_probs", "RpnRoiProbs"},
      {"rpn_rois", "RpnRois"},
      {"rpn_rois_num", "RpnRoisNum"},
      {"scores", "Scores"},
      {"variances", "Variances"},
    }},
    {"legacy_matmul", {
      {"out", "Out"},
      {"out_grad", "DOut"},
      {"x", "X"},
      {"x_grad", "DX"},
      {"x_grad_grad", "DDX"},
      {"y", "Y"},
      {"y_grad", "DY"},
      {"y_grad_grad", "DDY"},
    }},
    {"legacy_nearest_interp", {
      {"out_size", "OutSize"},
      {"output", "Out"},
      {"scale_tensor", "Scale"},
      {"size_tensor", "SizeTensor"},
      {"x", "X"},
    }},
    {"legacy_reshape", {
      {"out", "Out"},
      {"x", "X"},
      {"xshape", "XShape"},
    }},
    {"lerp", {
      {"out", "Out"},
      {"weight", "Weight"},
      {"x", "X"},
      {"y", "Y"},
    }},
    {"less_equal", {
      {"out", "Out"},
      {"x", "X"},
      {"y", "Y"},
    }},
    {"less_than", {
      {"out", "Out"},
      {"x", "X"},
      {"y", "Y"},
    }},
    {"lgamma", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"limit_by_capacity", {
      {"out", "Out"},
    }},
    {"linear_interp", {
      {"out_size", "OutSize"},
      {"output", "Out"},
      {"scale_tensor", "Scale"},
      {"size_tensor", "SizeTensor"},
      {"x", "X"},
    }},
    {"linspace", {
      {"number", "Num"},
      {"out", "Out"},
      {"start", "Start"},
      {"stop", "Stop"},
    }},
    {"lod_array_length", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"lod_reset", {
      {"out", "Out"},
      {"x", "X"},
      {"y", "Y"},
    }},
    {"log", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"log10", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"log1p", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"log2", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"log_loss", {
      {"input", "Predicted"},
      {"label", "Labels"},
      {"out", "Loss"},
    }},
    {"log_softmax", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"logcumsumexp", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"logical_and", {
      {"out", "Out"},
      {"x", "X"},
      {"y", "Y"},
    }},
    {"logical_not", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"logical_or", {
      {"out", "Out"},
      {"x", "X"},
      {"y", "Y"},
    }},
    {"logical_xor", {
      {"out", "Out"},
      {"x", "X"},
      {"y", "Y"},
    }},
    {"logit", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"logsigmoid", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"logspace", {
      {"base", "Base"},
      {"num", "Num"},
      {"out", "Out"},
      {"start", "Start"},
      {"stop", "Stop"},
    }},
    {"logsumexp", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"lookup_table", {
      {"ids", "Ids"},
      {"out", "Out"},
      {"w", "W"},
    }},
    {"lookup_table_dequant", {
      {"ids", "Ids"},
      {"out", "Out"},
      {"w", "W"},
    }},
    {"lrn", {
      {"mid_out", "MidOut"},
      {"out", "Out"},
      {"x", "X"},
    }},
    {"lstm", {
      {"batch_cell_pre_act", "BatchCellPreAct"},
      {"batch_gate", "BatchGate"},
      {"bias", "Bias"},
      {"c0", "C0"},
      {"cell", "Cell"},
      {"h0", "H0"},
      {"hidden", "Hidden"},
      {"input", "Input"},
      {"weight", "Weight"},
    }},
    {"lstsq", {
      {"rank", "Rank"},
      {"residuals", "Residuals"},
      {"singular_values", "SingularValues"},
      {"solution", "Solution"},
      {"x", "X"},
      {"y", "Y"},
    }},
    {"lu", {
      {"infos", "Infos"},
      {"out", "Out"},
      {"pivots", "Pivots"},
      {"x", "X"},
    }},
    {"lu_unpack", {
      {"l", "L"},
      {"pmat", "Pmat"},
      {"u", "U"},
      {"x", "X"},
      {"y", "Pivots"},
    }},
    {"margin_cross_entropy", {
      {"label", "Label"},
      {"logits", "Logits"},
      {"loss", "Loss"},
      {"softmax", "Softmax"},
    }},
    {"masked_select", {
      {"mask", "Mask"},
      {"out", "Y"},
      {"x", "X"},
    }},
    {"match_matrix_tensor", {
      {"out", "Out"},
      {"tmp", "Tmp"},
      {"w", "W"},
      {"x", "X"},
      {"y", "Y"},
    }},
    {"matmul", {
      {"out", "Out"},
      {"x", "X"},
      {"y", "Y"},
    }},
    {"matmul_with_flatten", {
      {"out", "Out"},
      {"x", "X"},
      {"y", "Y"},
    }},
    {"matrix_nms", {
      {"bboxes", "BBoxes"},
      {"index", "Index"},
      {"out", "Out"},
      {"roisnum", "RoisNum"},
      {"scores", "Scores"},
    }},
    {"matrix_power", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"matrix_rank", {
      {"out", "Out"},
      {"tol_tensor", "TolTensor"},
      {"x", "X"},
    }},
    {"max", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"max_pool2d_with_index", {
      {"mask", "Mask"},
      {"out", "Out"},
      {"x", "X"},
    }},
    {"max_pool3d_with_index", {
      {"mask", "Mask"},
      {"out", "Out"},
      {"x", "X"},
    }},
    {"maximum", {
      {"out", "Out"},
      {"x", "X"},
      {"y", "Y"},
    }},
    {"maxout", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"mean", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"mean_all", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"memcpy", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"memcpy_d2h", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"merge_selected_rows", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"merged_adam_", {
      {"beta1_pow", "Beta1Pow"},
      {"beta1_pow_out", "Beta1PowOut"},
      {"beta2_pow", "Beta2Pow"},
      {"beta2_pow_out", "Beta2PowOut"},
      {"grad", "Grad"},
      {"learning_rate", "LearningRate"},
      {"master_param", "MasterParam"},
      {"master_param_out", "MasterParamOut"},
      {"moment1", "Moment1"},
      {"moment1_out", "Moment1Out"},
      {"moment2", "Moment2"},
      {"moment2_max", "Moment2Max"},
      {"moment2_max_out", "Moment2MaxOut"},
      {"moment2_out", "Moment2Out"},
      {"param", "Param"},
      {"param_out", "ParamOut"},
    }},
    {"merged_momentum_", {
      {"grad", "Grad"},
      {"learning_rate", "LearningRate"},
      {"master_param", "MasterParam"},
      {"master_param_out", "MasterParamOut"},
      {"param", "Param"},
      {"param_out", "ParamOut"},
      {"velocity", "Velocity"},
      {"velocity_out", "VelocityOut"},
    }},
    {"meshgrid", {
      {"inputs", "X"},
      {"out", "Out"},
    }},
    {"min", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"minimum", {
      {"out", "Out"},
      {"x", "X"},
      {"y", "Y"},
    }},
    {"mish", {
      {"lambda", "threshold"},
      {"out", "Out"},
      {"x", "X"},
    }},
    {"mode", {
      {"indices", "Indices"},
      {"out", "Out"},
      {"x", "X"},
    }},
    {"momentum_", {
      {"grad", "Grad"},
      {"learning_rate", "LearningRate"},
      {"master_param", "MasterParam"},
      {"master_param_out", "MasterParamOut"},
      {"param", "Param"},
      {"param_out", "ParamOut"},
      {"velocity", "Velocity"},
      {"velocity_out", "VelocityOut"},
    }},
    {"moving_average_abs_max_scale", {
      {"in_accum", "InAccum"},
      {"in_state", "InState"},
      {"out", "Out"},
      {"out_accum", "OutAccum"},
      {"out_scale", "OutScale"},
      {"out_state", "OutState"},
      {"x", "X"},
    }},
    {"mp_allreduce_sum", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"multi_dot", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"multi_gru", {
      {"bias", "Bias"},
      {"hidden", "Hidden"},
      {"scale_weights", "Scale_weights"},
      {"weight_h", "WeightH"},
      {"weight_x", "WeightX"},
      {"x", "X"},
    }},
    {"multiclass_nms", {
      {"bboxes", "BBoxes"},
      {"out", "Out"},
      {"scores", "Scores"},
    }},
    {"multiclass_nms3", {
      {"bboxes", "BBoxes"},
      {"index", "Index"},
      {"nms_rois_num", "NmsRoisNum"},
      {"out", "Out"},
      {"rois_num", "RoisNum"},
      {"scores", "Scores"},
    }},
    {"multihead_matmul", {
      {"bias", "Bias"},
      {"bias_qk", "BiasQK"},
      {"input", "Input"},
      {"out", "Out"},
      {"w", "W"},
    }},
    {"multinomial", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"multiplex", {
      {"index", "Ids"},
      {"inputs", "X"},
      {"out", "Out"},
    }},
    {"multiply", {
      {"out", "Out"},
      {"x", "X"},
      {"y", "Y"},
    }},
    {"mv", {
      {"out", "Out"},
      {"vec", "Vec"},
      {"x", "X"},
    }},
    {"nanmedian", {
      {"medians", "MedianIndex"},
      {"out", "Out"},
      {"x", "X"},
    }},
    {"nce", {
      {"bias", "Bias"},
      {"cost", "Cost"},
      {"custom_dist_alias", "CustomDistAlias"},
      {"custom_dist_alias_probs", "CustomDistAliasProbs"},
      {"custom_dist_probs", "CustomDistProbs"},
      {"input", "Input"},
      {"label", "Label"},
      {"sample_labels", "SampleLabels"},
      {"sample_logits", "SampleLogits"},
      {"sample_weight", "SampleWeight"},
      {"weight", "Weight"},
    }},
    {"nearest_interp", {
      {"out_size", "OutSize"},
      {"output", "Out"},
      {"scale_tensor", "Scale"},
      {"size_tensor", "SizeTensor"},
      {"x", "X"},
    }},
    {"nll_loss", {
      {"input", "X"},
      {"label", "Label"},
      {"out", "Out"},
      {"total_weight", "Total_weight"},
      {"weight", "Weight"},
    }},
    {"nms", {
      {"out", "KeepBoxesIdxs"},
      {"x", "Boxes"},
    }},
    {"nonzero", {
      {"condition", "Condition"},
      {"out", "Out"},
    }},
    {"nop", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"norm", {
      {"norm", "Norm"},
      {"out", "Out"},
      {"x", "X"},
    }},
    {"not_equal", {
      {"out", "Out"},
      {"x", "X"},
      {"y", "Y"},
    }},
    {"number_count", {
      {"numbers", "numbers"},
      {"out", "Out"},
    }},
    {"numel", {
      {"size", "Out"},
      {"x", "Input"},
    }},
    {"one_hot", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"overlap_add", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"p_norm", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"pad", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"pad3d", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"partial_allgather", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"partial_concat", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"partial_recv", {
      {"out", "Out"},
    }},
    {"partial_send", {
      {"x", "X"},
    }},
    {"partial_sum", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"pixel_shuffle", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"pixel_unshuffle", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"poisson", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"pool2d", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"pool3d", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"pow", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"prelu", {
      {"alpha", "Alpha"},
      {"out", "Out"},
      {"x", "X"},
    }},
    {"print", {
      {"in", "In"},
      {"out", "Out"},
    }},
    {"prior_box", {
      {"image", "Image"},
      {"input", "Input"},
      {"out", "Boxes"},
      {"var", "Variances"},
    }},
    {"prod", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"prune_gate_by_capacity", {
      {"expert_count", "ExpertCount"},
      {"gate_idx", "GateIdx"},
      {"out_gate_idx", "NewGateIdx"},
    }},
    {"psroi_pool", {
      {"boxes", "ROIs"},
      {"boxes_num", "RoisNum"},
      {"out", "Out"},
      {"x", "X"},
    }},
    {"put_along_axis", {
      {"arr", "Input"},
      {"indices", "Index"},
      {"out", "Result"},
      {"values", "Value"},
    }},
    {"pyramid_hash", {
      {"black_list", "BlackList"},
      {"drop_pos", "DropPos"},
      {"out", "Out"},
      {"w", "W"},
      {"white_list", "WhiteList"},
      {"x", "X"},
      {"x_temp_out", "X_Temp_Out"},
    }},
    {"qr", {
      {"q", "Q"},
      {"r", "R"},
      {"x", "X"},
    }},
    {"quantize", {
      {"input", "Input"},
      {"output", "Output"},
    }},
    {"quantize_linear", {
      {"in_accum", "InAccum"},
      {"in_state", "InState"},
      {"out_accum", "OutAccum"},
      {"out_scale", "OutScale"},
      {"out_state", "OutState"},
      {"scale", "Scale"},
      {"x", "X"},
      {"y", "Y"},
      {"zero_point", "ZeroPoint"},
    }},
    {"randint", {
      {"out", "Out"},
    }},
    {"random_routing", {
      {"out", "Out"},
      {"prob", "Prob"},
      {"topk_idx", "TopK_Idx"},
      {"topk_value", "TopK_Value"},
    }},
    {"randperm", {
      {"out", "Out"},
    }},
    {"range_v2", {
      {"end", "End"},
      {"out", "Out"},
      {"start", "Start"},
      {"step", "Step"},
    }},
    {"rank_attention", {
      {"input_help", "InputHelp"},
      {"ins_rank", "InsRank"},
      {"out", "Out"},
      {"rank_offset", "RankOffset"},
      {"rank_param", "RankParam"},
      {"x", "X"},
    }},
    {"read_from_array", {
      {"array", "X"},
      {"i", "I"},
      {"out", "Out"},
    }},
    {"real", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"reciprocal", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"recv_v2", {
      {"out", "Out"},
    }},
    {"reindex_graph", {
      {"count", "Count"},
      {"hashtable_index", "HashTable_Index"},
      {"hashtable_value", "HashTable_Value"},
      {"neighbors", "Neighbors"},
      {"out_nodes", "Out_Nodes"},
      {"reindex_dst", "Reindex_Dst"},
      {"reindex_src", "Reindex_Src"},
      {"x", "X"},
    }},
    {"relu", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"relu6", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"remainder", {
      {"out", "Out"},
      {"x", "X"},
      {"y", "Y"},
    }},
    {"renorm", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"repeat_interleave", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"repeat_interleave_with_tensor_index", {
      {"out", "Out"},
      {"repeats", "RepeatTensor"},
      {"x", "X"},
    }},
    {"requantize", {
      {"input", "Input"},
      {"output", "Output"},
    }},
    {"reshape", {
      {"out", "Out"},
      {"x", "X"},
      {"xshape", "XShape"},
    }},
    {"resnet_basic_block", {
      {"bias1", "Bias1"},
      {"bias2", "Bias2"},
      {"bias3", "Bias3"},
      {"conv1", "Conv1"},
      {"conv2", "Conv2"},
      {"conv2_input", "Conv2Input"},
      {"conv3", "Conv3"},
      {"filter1", "Filter1"},
      {"filter2", "Filter2"},
      {"filter3", "Filter3"},
      {"max_filter1", "MaxFilter1"},
      {"max_filter2", "MaxFilter2"},
      {"max_filter3", "MaxFilter3"},
      {"max_input1", "MaxInput1"},
      {"max_input2", "MaxInput2"},
      {"max_input3", "MaxInput3"},
      {"mean1", "Mean1"},
      {"mean1_out", "Mean1Out"},
      {"mean2", "Mean2"},
      {"mean2_out", "Mean2Out"},
      {"mean3", "Mean3"},
      {"mean3_out", "Mean3Out"},
      {"out", "Y"},
      {"saved_invstd1", "SavedInvstd1"},
      {"saved_invstd2", "SavedInvstd2"},
      {"saved_invstd3", "SavedInvstd3"},
      {"saved_mean1", "SavedMean1"},
      {"saved_mean2", "SavedMean2"},
      {"saved_mean3", "SavedMean3"},
      {"scale1", "Scale1"},
      {"scale2", "Scale2"},
      {"scale3", "Scale3"},
      {"var1", "Var1"},
      {"var1_out", "Var1Out"},
      {"var2", "Var2"},
      {"var2_out", "Var2Out"},
      {"var3", "Var3"},
      {"var3_out", "Var3Out"},
      {"x", "X"},
    }},
    {"resnet_unit", {
      {"bias_x", "BiasX"},
      {"bias_z", "BiasZ"},
      {"bit_mask", "BitMask"},
      {"conv_x", "ConvX"},
      {"conv_z", "ConvZ"},
      {"filter_x", "FilterX"},
      {"filter_z", "FilterZ"},
      {"mean_x", "MeanX"},
      {"mean_z", "MeanZ"},
      {"out", "Y"},
      {"running_mean_x", "RunningMeanX"},
      {"running_mean_z", "RunningMeanZ"},
      {"running_var_x", "RunningVarX"},
      {"running_var_z", "RunningVarZ"},
      {"saved_invstd_x", "SavedInvstdX"},
      {"saved_invstd_z", "SavedInvstdZ"},
      {"saved_mean_x", "SavedMeanX"},
      {"saved_mean_z", "SavedMeanZ"},
      {"scale_x", "ScaleX"},
      {"scale_z", "ScaleZ"},
      {"var_x", "VarX"},
      {"var_z", "VarZ"},
      {"x", "X"},
      {"z", "Z"},
    }},
    {"reverse", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"rmsprop_", {
      {"grad", "Grad"},
      {"learning_rate", "LearningRate"},
      {"master_param", "MasterParam"},
      {"master_param_outs", "MasterParamOut"},
      {"mean_grad", "MeanGrad"},
      {"mean_grad_out", "MeanGradOut"},
      {"mean_square", "MeanSquare"},
      {"mean_square_out", "MeanSquareOut"},
      {"moment", "Moment"},
      {"moment_out", "MomentOut"},
      {"param", "Param"},
      {"param_out", "ParamOut"},
    }},
    {"rnn", {
      {"dropout_state_out", "DropoutState"},
      {"out", "Out"},
      {"pre_state", "PreState"},
      {"reserve", "Reserve"},
      {"sequence_length", "SequenceLength"},
      {"state", "State"},
      {"weight_list", "WeightList"},
      {"x", "Input"},
    }},
    {"roi_align", {
      {"boxes", "ROIs"},
      {"boxes_num", "RoisNum"},
      {"out", "Out"},
      {"x", "X"},
    }},
    {"roi_pool", {
      {"arg_max", "Argmax"},
      {"boxes", "ROIs"},
      {"boxes_num", "RoisNum"},
      {"out", "Out"},
      {"x", "X"},
    }},
    {"roll", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"round", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"row_conv", {
      {"filter", "Filter"},
      {"out", "Out"},
      {"x", "X"},
    }},
    {"rrelu", {
      {"noise", "Noise"},
      {"out", "Out"},
      {"x", "X"},
    }},
    {"rsqrt", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"save_combine", {
      {"x", "X"},
    }},
    {"scale", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"scatter", {
      {"index", "Ids"},
      {"out", "Out"},
      {"updates", "Updates"},
      {"x", "X"},
    }},
    {"scatter_nd_add", {
      {"index", "Index"},
      {"out", "Out"},
      {"updates", "Updates"},
      {"x", "X"},
    }},
    {"searchsorted", {
      {"out", "Out"},
      {"sorted_sequence", "SortedSequence"},
      {"values", "Values"},
    }},
    {"seed", {
      {"out", "Out"},
    }},
    {"segment_pool", {
      {"out", "Out"},
      {"segment_ids", "SegmentIds"},
      {"summed_ids", "SummedIds"},
      {"x", "X"},
    }},
    {"self_dp_attention", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"selu", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"send_u_recv", {
      {"dst_count", "Dst_count"},
      {"dst_index", "Dst_index"},
      {"out", "Out"},
      {"src_index", "Src_index"},
      {"x", "X"},
    }},
    {"send_ue_recv", {
      {"dst_count", "Dst_count"},
      {"dst_index", "Dst_index"},
      {"out", "Out"},
      {"src_index", "Src_index"},
      {"x", "X"},
      {"y", "Y"},
    }},
    {"send_v2", {
      {"x", "X"},
    }},
    {"sequence_conv", {
      {"filter", "Filter"},
      {"out", "Out"},
      {"padding_data", "PaddingData"},
      {"x", "X"},
    }},
    {"sequence_expand", {
      {"out", "Out"},
      {"x", "X"},
      {"y", "Y"},
    }},
    {"sequence_mask", {
      {"x", "X"},
      {"y", "Y"},
    }},
    {"sequence_pool", {
      {"max_index", "MaxIndex"},
      {"out", "Out"},
      {"x", "X"},
    }},
    {"sequence_softmax", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"set_value", {
      {"out", "Out"},
      {"x", "Input"},
    }},
    {"set_value_with_tensor", {
      {"out", "Out"},
      {"x", "Input"},
    }},
    {"sgd_", {
      {"grad", "Grad"},
      {"learning_rate", "LearningRate"},
      {"master_param", "MasterParam"},
      {"master_param_out", "MasterParamOut"},
      {"param", "Param"},
      {"param_out", "ParamOut"},
    }},
    {"shape", {
      {"input", "Input"},
      {"out", "Out"},
    }},
    {"shard_index", {
      {"input", "X"},
      {"out", "Out"},
    }},
    {"share_buffer", {
      {"out", "Out"},
      {"x", "X"},
      {"xout", "XOut"},
    }},
    {"share_data", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"share_data_", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"shuffle_batch", {
      {"out", "Out"},
      {"seed", "Seed"},
      {"seed_out", "SeedOut"},
      {"shuffle_idx", "ShuffleIdx"},
      {"x", "X"},
    }},
    {"shuffle_channel", {
      {"group", "group"},
      {"out", "Out"},
      {"x", "X"},
    }},
    {"sigmoid", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"sigmoid_cross_entropy_with_logits", {
      {"label", "Label"},
      {"out", "Out"},
      {"x", "X"},
    }},
    {"sign", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"silu", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"sin", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"sinh", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"skip_layernorm", {
      {"bias", "Bias"},
      {"out", "Out"},
      {"scale", "Scale"},
      {"x", "X"},
      {"y", "Y"},
    }},
    {"slice", {
      {"input", "Input"},
      {"out", "Out"},
    }},
    {"slogdet", {
      {"out", "Out"},
      {"x", "Input"},
    }},
    {"soft_relu", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"softmax", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"softplus", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"softshrink", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"softsign", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"solve", {
      {"out", "Out"},
      {"x", "X"},
      {"y", "Y"},
    }},
    {"sparse_attention", {
      {"attn_mask", "AttnMask"},
      {"columns", "Columns"},
      {"k", "K"},
      {"key_padding_mask", "KeyPaddingMask"},
      {"offset", "Offset"},
      {"out", "Out"},
      {"q", "Q"},
      {"softmax", "Softmax"},
      {"sparse_dot_sdd", "SparseDotSdd"},
      {"v", "V"},
    }},
    {"sparse_momentum", {
      {"axis", "Axis"},
      {"grad", "Grad"},
      {"index", "Index"},
      {"learning_rate", "LearningRate"},
      {"master_param", "MasterParam"},
      {"master_param_out", "MasterParamOut"},
      {"param", "Param"},
      {"param_out", "ParamOut"},
      {"velocity", "Velocity"},
      {"velocity_out", "VelocityOut"},
    }},
    {"spectral_norm", {
      {"out", "Out"},
      {"u", "U"},
      {"v", "V"},
      {"weight", "Weight"},
    }},
    {"split", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"sqrt", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"square", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"squared_l2_norm", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"squeeze", {
      {"out", "Out"},
      {"x", "X"},
      {"xshape", "XShape"},
    }},
    {"stack", {
      {"out", "Y"},
      {"x", "X"},
    }},
    {"stanh", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"stft", {
      {"out", "Out"},
      {"window", "Window"},
      {"x", "X"},
    }},
    {"straight_through_estimator_grad", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"strided_slice", {
      {"out", "Out"},
      {"x", "Input"},
    }},
    {"subtract", {
      {"out", "Out"},
      {"x", "X"},
      {"y", "Y"},
    }},
    {"sum", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"svd", {
      {"s", "S"},
      {"u", "U"},
      {"vh", "VH"},
      {"x", "X"},
    }},
    {"swish", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"sync_batch_norm", {
      {"bias", "Bias"},
      {"mean", "Mean"},
      {"mean_out", "MeanOut"},
      {"out", "Y"},
      {"reserve_space", "ReserveSpace"},
      {"saved_mean", "SavedMean"},
      {"saved_variance", "SavedVariance"},
      {"scale", "Scale"},
      {"variance", "Variance"},
      {"variance_out", "VarianceOut"},
      {"x", "X"},
    }},
    {"sync_calc_stream", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"sync_comm_stream", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"take_along_axis", {
      {"arr", "Input"},
      {"indices", "Index"},
      {"out", "Result"},
    }},
    {"tan", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"tanh", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"tanh_shrink", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"tdm_child", {
      {"child", "Child"},
      {"child_nums", "child_nums"},
      {"dtype", "dtype"},
      {"leaf_mask", "LeafMask"},
      {"tree_info", "TreeInfo"},
      {"x", "X"},
    }},
    {"tdm_sampler", {
      {"labels", "Labels"},
      {"layer", "Layer"},
      {"mask", "Mask"},
      {"out", "Out"},
      {"travel", "Travel"},
      {"x", "X"},
    }},
    {"temporal_shift", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"thresholded_relu", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"tile", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"topk", {
      {"indices", "Indices"},
      {"out", "Out"},
      {"x", "X"},
    }},
    {"topk_v1", {
      {"indices", "Indices"},
      {"out", "Out"},
      {"x", "X"},
    }},
    {"trace", {
      {"out", "Out"},
      {"x", "Input"},
    }},
    {"transfer_layout", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"transpose", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"triangular_solve", {
      {"out", "Out"},
      {"x", "X"},
      {"y", "Y"},
    }},
    {"tril_triu", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"trilinear_interp", {
      {"out_size", "OutSize"},
      {"output", "Out"},
      {"scale_tensor", "Scale"},
      {"size_tensor", "SizeTensor"},
      {"x", "X"},
    }},
    {"trunc", {
      {"input", "X"},
      {"out", "Out"},
    }},
    {"truncated_gaussian_random", {
      {"out", "Out"},
    }},
    {"unbind", {
      {"input", "X"},
      {"out", "Out"},
    }},
    {"unfold", {
      {"out", "Y"},
      {"x", "X"},
    }},
    {"uniform", {
      {"out", "Out"},
    }},
    {"uniform_inplace", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"uniform_random_batch_size_like", {
      {"input", "Input"},
      {"out", "Out"},
    }},
    {"unique", {
      {"counts", "Counts"},
      {"indices", "Indices"},
      {"inverse", "Index"},
      {"out", "Out"},
      {"x", "X"},
    }},
    {"unique_consecutive", {
      {"counts", "Counts"},
      {"index", "Index"},
      {"out", "Out"},
      {"x", "X"},
    }},
    {"unpool", {
      {"indices", "Indices"},
      {"out", "Out"},
      {"x", "X"},
    }},
    {"unpool3d", {
      {"indices", "Indices"},
      {"out", "Out"},
      {"x", "X"},
    }},
    {"unsqueeze", {
      {"out", "Out"},
      {"x", "X"},
      {"xshape", "XShape"},
    }},
    {"unstack", {
      {"out", "Y"},
      {"x", "X"},
    }},
    {"update_loss_scaling_", {
      {"found_infinite", "FoundInfinite"},
      {"in_bad_steps", "InBadSteps"},
      {"in_good_steps", "InGoodSteps"},
      {"loss_scaling", "LossScaling"},
      {"out", "Out"},
      {"out_bad_steps", "OutBadSteps"},
      {"out_good_steps", "OutGoodSteps"},
      {"prev_loss_scaling", "PrevLossScaling"},
      {"x", "X"},
    }},
    {"viterbi_decode", {
      {"lengths", "Length"},
      {"path", "Path"},
      {"potentials", "Input"},
      {"scores", "Scores"},
      {"transition_params", "Transition"},
    }},
    {"warpctc", {
      {"label", "Label"},
      {"labels_length", "LabelLength"},
      {"logits", "Logits"},
      {"logits_length", "LogitsLength"},
      {"loss", "Loss"},
      {"warpctcgrad", "WarpCTCGrad"},
    }},
    {"where", {
      {"condition", "Condition"},
      {"out", "Out"},
      {"x", "X"},
      {"y", "Y"},
    }},
    {"write_to_array", {
      {"i", "I"},
      {"out", "Out"},
      {"x", "X"},
    }},
    {"yolo_box", {
      {"boxes", "Boxes"},
      {"img_size", "ImgSize"},
      {"scores", "Scores"},
      {"x", "X"},
    }},
    {"yolo_box_head", {
      {"out", "Out"},
      {"x", "X"},
    }},
    {"yolo_box_post", {
      {"boxes0", "Boxes0"},
      {"boxes1", "Boxes1"},
      {"boxes2", "Boxes2"},
      {"image_scale", "ImageScale"},
      {"image_shape", "ImageShape"},
      {"nms_rois_num", "NmsRoisNum"},
      {"out", "Out"},
    }},
    {"yolo_loss", {
      {"gt_box", "GTBox"},
      {"gt_label", "GTLabel"},
      {"gt_match_mask", "GTMatchMask"},
      {"gt_score", "GTScore"},
      {"loss", "Loss"},
      {"objectness_mask", "ObjectnessMask"},
      {"x", "X"},
    }},
  };
  static const std::unordered_map<std::string, std::string> empty;
  auto it = all.find(op_name);
  return it != all.end() ? it->second : empty;
}

}  // namespace pir
}  // namespace paddle2onnx
