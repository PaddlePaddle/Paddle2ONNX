
from auto_scan_test import OPConvertAutoScanTest, BaseNet
from hypothesis import reproduce_failure
import hypothesis.strategies as st
from onnxbase import randtool
import numpy as np
import unittest
import paddle


class Net(BaseNet):
    def forward(self):
        shape = self.config["shape"]
        # todo tensor is not supported
        if self.config['is_shape_tensor']:
            shape = paddle.to_tensor(shape).astype(self.config['shape_dtype'])
        dtype = self.config["dtype"]
        x = paddle.empty(shape, dtype=dtype)
        return x


class TestFullConvert(OPConvertAutoScanTest):
    """
    api: paddle.full
    OPset version:
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=2, max_value=20), min_size=0, max_size=4))
        dtype = draw(st.sampled_from(["float32", "float64", "int32", "int64", "bool"]))
        shape_dtype = draw(st.sampled_from(["int32", "int64"]))
        # todo tensor is not supported
        is_tensor = False  # draw(st.booleans())
        is_shape_tensor = draw(st.booleans())
        print("===dtype===", dtype)
        print("===shape_dtype", shape_dtype)
        print("===is_shape_tensor===", is_shape_tensor)
        # if is_shape_tensor:
        #     opset_version = [9, 11, 15]
        # else:
        #     opset_version = [7, 9, 15]
        opset_version = [11]

        config = {
            "op_names": ["fill_constant"],
            "test_data_shapes": [],
            "test_data_types": [],
            "opset_version": opset_version,
            "input_spec_shape": [],
            "shape": input_shape,
            "dtype": dtype,
            "is_tensor": is_tensor,
            "is_shape_tensor": is_shape_tensor,
            "shape_dtype": shape_dtype,
        }

        model = Net(config)

        return (config, model)

    def test(self):
        self.run_and_statis(max_examples=30, max_duration=-1)


# class Net1(BaseNet):
#     def forward(self):
#         fill_value = self.config['fill_value']
#         shape = [
#             2, 1, paddle.to_tensor(
#                 2, dtype=self.config['shape_dtype']), 3, 2, 2
#         ]
#         # TODO not supported
#         # shape = [paddle.to_tensor(2), paddle.to_tensor(np.array(1).astype("int64")), 2, 3, 2, 2]
#         dtype = self.config["dtype"]
#         x = paddle.full(shape=shape, fill_value=fill_value, dtype=dtype)
#         print("=======================\n",x.shape)
#         return x


# class TestFullConvert1(OPConvertAutoScanTest):
#     """
#     api: paddle.full
#     OPset version:
#     """

#     def sample_convert_config(self, draw):
#         input_shape = draw(
#             st.lists(
#                 st.integers(
#                     min_value=2, max_value=20), min_size=1, max_size=4))
#         dtype = draw(st.sampled_from(["float32", "float64", "int32", "int64","bool"]))
#         shape_dtype = draw(st.sampled_from(["int32", "int64"]))

#         fill_value = draw(st.integers(min_value=1, max_value=5))
#         # todo tensor is not supported
#         is_tensor = False  # draw(st.booleans())
#         is_shape_tensor = True  # draw(st.booleans())
#         if is_shape_tensor:
#             opset_version = [9, 15]
#         else:
#             opset_version = [7, 9, 15]
#         config = {
#             "op_names": ["fill_constant"],
#             "test_data_shapes": [],
#             "test_data_types": [],
#             "opset_version": opset_version,
#             "input_spec_shape": [],
#             "shape": input_shape,
#             "dtype": dtype,
#             "fill_value": fill_value,
#             "shape_dtype": shape_dtype,
#         }

#         model = Net1(config)

#         return (config, model)

#     def test(self):
#         self.run_and_statis(max_examples=30, max_duration=-1)


# class Net2(BaseNet):
#     def forward(self):
#         fill_value = self.config['fill_value']
#         shape = [paddle.to_tensor(1, dtype="int64")]
#         if self.config['is_tensor']:
#             fill_value = paddle.to_tensor(fill_value, dtype="int64")
#         # TODO not supported
#         # shape = [paddle.to_tensor(2), paddle.to_tensor(np.array(1).astype("int64")), 2, 3, 2, 2]
#         dtype = self.config["dtype"]
#         x = paddle.full(shape=shape, fill_value=fill_value, dtype=dtype)
#         return x


# class TestFullConvert2(OPConvertAutoScanTest):
#     """
#     api: paddle.full
#     OPset version:
#     """

#     def sample_convert_config(self, draw):
#         input_shape = draw(
#             st.lists(
#                 st.integers(
#                     min_value=2, max_value=20), min_size=1, max_size=4))
#         dtype = draw(st.sampled_from(["float32", "float64", "int32", "int64","bool"]))
#         shape_dtype = draw(st.sampled_from(["int32", "int64"]))

#         fill_value = draw(st.integers(min_value=1, max_value=5))
#         # todo tensor is not supported
#         is_tensor = draw(st.booleans())
#         is_shape_tensor = True  # draw(st.booleans())
#         if is_shape_tensor:
#             opset_version = [9, 11, 15]
#         else:
#             opset_version = [7, 9, 15]
#         config = {
#             "op_names": ["fill_constant"],
#             "test_data_shapes": [],
#             "test_data_types": [],
#             "opset_version": opset_version,
#             "input_spec_shape": [],
#             "shape": input_shape,
#             "dtype": dtype,
#             "fill_value": fill_value,
#             "shape_dtype": shape_dtype,
#             "is_tensor": is_tensor,
#         }

#         model = Net2(config)

#         return (config, model)

#     def test(self):
#         self.run_and_statis(max_examples=30, max_duration=-1)


if __name__ == "__main__":
    unittest.main()
