import argparse

from onnx import *
import paddle.fluid as fluid

# import ops

def convert(dirname=''):
    # Read the model files.
    # place = fluid.CPUPlace()
    # exe = fluid.Executor(place)

    # inference_scope = fluid.core.Scope()
    # with fluid.scope_guard(inference_scope):
    #   [inference_program, feed_target_names,
    #       fetch_targets] = fluid.io.load_inference_model(dirname, exe)

    # Using blocks in programs, create nodes using:
    # helper.make_program
    # helper.make_node
    # ops.PADDLE_TO_ONNX


if __name__ == "__main__":
    # Read arguments: path to model.
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--modeldir", required=True, help="input model")

    convert()
