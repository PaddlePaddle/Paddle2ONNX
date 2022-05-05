import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import sys
import os
import copy
import numpy as np

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
EXPLICIT_PRECISION = 1 << (
    int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_PRECISION)


def remove_initializer_from_input(ori_model):
    model = copy.deepcopy(ori_model)
    if model.ir_version < 4:
        print(
            'Model with ir_version below 4 requires to include initilizer in graph input'
        )
        return

    inputs = model.graph.input
    name_to_input = {}
    for input in inputs:
        name_to_input[input.name] = input

    for initializer in model.graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])
    return model


# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem
        if host_mem:
            self.nbytes = host_mem.nbytes
        else:
            self.nbytes = 0

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class TrtEngine:
    def __init__(self,
                 onnx_model_file,
                 shape_info=None,
                 max_batch_size=None,
                 use_int8=False,
                 engine_file_path=None):
        self.max_batch_size = 1 if max_batch_size is None else max_batch_size
        TRT_LOGGER = trt.Logger()
        if engine_file_path is not None and os.path.exists(engine_file_path):
            # If a serialized engine exists, use it instead of building an engine.
            print("Reading engine from file {}".format(engine_file_path))
            with open(engine_file_path,
                      "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
        else:
            builder = trt.Builder(TRT_LOGGER)
            config = builder.create_builder_config()
            if use_int8:
                network = builder.create_network(EXPLICIT_BATCH |
                                                 EXPLICIT_PRECISION)
            else:
                network = builder.create_network(EXPLICIT_BATCH)
            parser = trt.OnnxParser(network, TRT_LOGGER)
            runtime = trt.Runtime(TRT_LOGGER)
            config.max_workspace_size = 1 << 28
            if use_int8:
                config.set_flag(trt.BuilderFlag.INT8)

            import onnx
            print('Loading ONNX model...')
            onnx_model = onnx_model_file
            print("=========type", type(onnx_model))
            if not isinstance(onnx_model_file, onnx.ModelProto):
                onnx_model = onnx.load(onnx_model_file)
            onnx_model = remove_initializer_from_input(onnx_model)
            if not parser.parse(onnx_model.SerializeToString()):
                print("ERROR: Failed to parse the ONNX file.")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                sys.exit(0)

            if shape_info is None:
                builder.max_batch_size = 1
                for i in range(len(onnx_model.graph.input)):
                    input_shape = [
                        x.dim_value
                        for x in onnx_model.graph.input[0]
                        .type.tensor_type.shape.dim
                    ]
                    for s in input_shape:
                        assert s > 0, "In static shape mode, the input of onnx model should be fixed, but now it's {}".format(
                            onnx_model.graph.input[i])
            else:
                max_batch_size = 1
                if shape_info is not None:
                    assert len(
                        shape_info
                    ) == network.num_inputs, "Length of shape_info: {} is not same with length of model input: {}".format(
                        len(shape_info), network.num_inputs)
                    profile = builder.create_optimization_profile()
                    for k, v in shape_info.items():
                        if v[2][0] > max_batch_size:
                            max_batch_size = v[2][0]
                        print("optimize shape", k, v[0], v[1], v[2])
                        profile.set_shape(k, v[0], v[1], v[2])
                    config.add_optimization_profile(profile)
                if max_batch_size > self.max_batch_size:
                    self.max_batch_size = max_batch_size
                builder.max_batch_size = self.max_batch_size

            print('Completed parsing of ONNX file')
            print('Building an engine from onnx model may take a while...')
            plan = builder.build_serialized_network(network, config)
            print("Engine!")
            self.engine = runtime.deserialize_cuda_engine(plan)
            print("Completed creating Engine")
            if engine_file_path is not None:
                with open(engine_file_path, "wb") as f:
                    f.write(self.engine.serialize())

        self.context = self.engine.create_execution_context()
        if shape_info is not None:
            self.context.active_optimization_profile = 0
        self.stream = cuda.Stream()
        self.bindings = []
        self.inputs = []
        self.outputs = []
        for binding in self.engine:
            self.bindings.append(0)
            if self.engine.binding_is_input(binding):
                self.inputs.append(HostDeviceMem(None, None))
            else:
                self.outputs.append(HostDeviceMem(None, None))

        print("Completed TrtEngine init ...")

    def infer(self, input_data):
        assert len(self.inputs) == len(
            input_data
        ), "Length of input_data: {} is not same with length of input: {}".format(
            len(input_data), len(self.inputs))

        self.allocate_buffers(input_data)

        return self.do_inference_v2(
            self.context,
            bindings=self.bindings,
            inputs=self.inputs,
            outputs=self.outputs,
            stream=self.stream)

    def do_inference_v2(self, context, bindings, inputs, outputs, stream):
        # Transfer input data to the GPU.
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        # Run inference.
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        [
            cuda.memcpy_dtoh_async(out.host, out.device, stream)
            for out in outputs
        ]
        # Synchronize the stream
        stream.synchronize()
        # Return only the host outputs.
        return [out.host for out in outputs]

    def allocate_buffers(self, input_data):
        input_idx = 0
        output_idx = 0
        for binding in self.engine:
            idx = self.engine.get_binding_index(binding)
            if self.engine.binding_is_input(binding):
                if not input_data[input_idx].flags['C_CONTIGUOUS']:
                    input_data[input_idx] = np.ascontiguousarray(input_data[
                        input_idx])
                self.context.set_binding_shape(idx,
                                               (input_data[input_idx].shape))
                self.inputs[input_idx].host = input_data[input_idx]
                nbytes = input_data[input_idx].nbytes
                if self.inputs[input_idx].nbytes < nbytes:
                    self.inputs[input_idx].nbytes = nbytes
                    self.inputs[input_idx].device = cuda.mem_alloc(nbytes)
                    self.bindings[idx] = int(self.inputs[input_idx].device)
                input_idx += 1
            else:
                dtype = trt.nptype(self.engine.get_binding_dtype(binding))
                shape = self.context.get_binding_shape(idx)
                self.outputs[output_idx].host = np.ascontiguousarray(
                    np.empty(
                        shape, dtype=dtype))
                nbytes = self.outputs[output_idx].host.nbytes
                if self.outputs[output_idx].nbytes < nbytes:
                    self.outputs[output_idx].nbytes = nbytes
                    self.outputs[output_idx].device = cuda.mem_alloc(
                        self.outputs[output_idx].host.nbytes)
                    self.bindings[idx] = int(self.outputs[output_idx].device)
                output_idx += 1


#
#def main():
#    import numpy as np
#    import onnxruntime as rt
#    onnx_model = 'resnet50.onnx'
#
#    # ONNXRuntime
#    sess = rt.InferenceSession(onnx_model)
#    input_name = sess.get_inputs()[0].name
#    label_name = sess.get_outputs()[0].name
#
#    trt_engine = TrtEngine(onnx_model_file=onnx_model, shape_info={0:[[1, 3, 224, 224], [5, 3, 224, 224], [10, 3, 224, 224]]}, engine_file_path='model.trt')
#
#    for i in range(10, 1, -1):
#        batch_size = i
#        data = np.array(np.random.randn(batch_size,3,224,224)).astype('float32')
#        trt_outputs = trt_engine.infer([data])
#
#        input_dict = {}
#        input_dict[sess.get_inputs()[0].name] = data
#        onnx_pred = sess.run(None, input_dict)[0]
#
#        diff = np.array(onnx_pred) - trt_outputs[0][0:batch_size * 1000].reshape(onnx_pred.shape)
#        diff = abs(diff)
#        print("Input index:",i,  ", max diff: ", np.amax(diff))
#
#
#if __name__ == "__main__":
#    main()
