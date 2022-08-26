import time
import numpy as np


class RKNNConfigPC:
    def __init__(self,
                 mean_values=None,
                 std_values=None,
                 model_path=None,
                 target='RK3588',
                 verbose=False,
                 export_path=None,
                 do_quantization=False,
                 outputs=None):
        from rknn.api import RKNN
        self.model_path = model_path
        self.target = target
        if mean_values is None:
            self.mean_values = [[0, 0, 0]]
        else:
            self.mean_values = mean_values
        if std_values is None:
            self.std_values = [[1, 1, 1]]
        else:
            self.std_values = std_values

        # create rknn
        self.rknn = RKNN(verbose)

        # pre-process config
        self.rknn.config(mean_values=self.mean_values,
                         std_values=self.std_values,
                         target_platform=self.target)

        # Load ONNX model
        if outputs is None:
            ret = self.rknn.load_onnx(model=self.model_path)
        else:
            ret = self.rknn.load_onnx(model=self.model_path,outputs=outputs)
        if ret != 0:
            print('【RKNNConfig】error :Load model failed!')
            exit(ret)

        # Build model
        ret = self.rknn.build(do_quantization=do_quantization, dataset="./images/coco/dataset.txt")
        if ret != 0:
            print('【RKNNConfig】error :Build model failed!')
            exit(ret)

        self.export_rknn(export_path)

        ret = self.rknn.init_runtime()
        if ret != 0:
            print('【RKNNConfig】error :Init runtime environment failed!')
            exit(ret)

    def export_rknn(self, export_path):
        if export_path is None:
            import os
            export_path = os.path.dirname(self.model_path) + "/../rknn/" + os.path.basename(self.model_path)[
                                                                           0:-5] + ".rknn"

        ret = self.rknn.export_rknn(export_path)
        if ret != 0:
            print('RKNNConfig】error : Export rknn model failed!')
            exit(ret)

    def infer(self, data):
        # 推理
        start_time = time.time()
        results = self.rknn.inference([data])
        end_time = time.time()
        return results, end_time - start_time

    def release(self):
        self.rknn.release()


class RKNNConfigBoard:
    def __init__(self,
                 rknn_path=None,
                 target='RK3588',
                 verbose=False):
        from rknnlite.api import RKNNLite
        if rknn_path is None:
            print("【RKNNConfig】error: rknn_path is None")
            exit(0)
        else:
            print("【RKNNConfig】: rknn will load by rknn")
            self.model_path = rknn_path
        self.target = target

        # create rknn
        self.rknn = RKNNLite(verbose=verbose)

        # Load ONNX model
        ret = self.rknn.load_rknn(path=self.model_path)
        if ret != 0:
            print('【RKNNConfig】error :Load model failed!')
            exit(ret)
        if self.target == "RK3588":
            ret = self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_1)
        else:
            ret = self.rknn.init_runtime()
        if ret != 0:
            print('【RKNNConfig】error :Init runtime environment failed!')
            exit(ret)

    def infer(self, data):
        # 推理
        start_time = time.time()
        results = self.rknn.inference([data])
        end_time = time.time()
        return results, end_time - start_time

    def release(self):
        self.rknn.release()
