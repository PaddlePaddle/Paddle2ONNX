class RKNNConfigPC:
    def __init__(self, onnx_path=None, target='RK3588'):
        self.model_path = onnx_path
        self.target = target

    def create_rknn(self, verbose=False, do_quantization=False, dataset=None, need_export=True,
                    export_path=None):
        from rknn.api import RKNN
        # create rknn
        self.rknn = RKNN(verbose)
        # pre-process config
        self.rknn.config(mean_values=[[0, 0, 0]], std_values=[[1, 1, 1]], target_platform=self.target)

        # Load ONNX model
        ret = self.rknn.load_onnx(model=self.model_path,outputs=['bilinear_interp_v2_2.tmp_0'])
        if ret != 0:
            print('【RKNNConfig】error :Load model failed!')
            exit(ret)

        # Build model
        ret = self.rknn.build(do_quantization=do_quantization, dataset=dataset)
        if ret != 0:
            print('【RKNNConfig】error :Build model failed!')
            exit(ret)

        if need_export:
            print("need export")
            self.export_rknn(export_path)
        ret = self.rknn.init_runtime()
        if ret != 0:
            print('【RKNNConfig】error :Init runtime environment failed!')
            exit(ret)
        return self.rknn

    def export_rknn(self, export_path):
        if export_path is None:
            import os
            export_path = os.path.dirname(self.model_path) + "/../rknn/" + os.path.basename(self.model_path)[
                                                                           0:-5] + ".rknn"
            # print(export_path)
        ret = self.rknn.export_rknn(export_path)
        if ret != 0:
            print('RKNNConfig】error : Export rknn model failed!')
            exit(ret)


class RKNNConfigBoard:
    def __init__(self, rknn_path=None, target='RK3588'):
        if rknn_path is None:
            print("【RKNNConfig】error: rknn_path is None")
            exit(0)
        else:
            print("【RKNNConfig】: rknn will load by rknn")
            self.model_path = rknn_path
        self.target = target

    def create_rknn(self, verbose=True):
        from rknnlite.api import RKNNLite
        # create rknn
        rknn = RKNNLite(verbose=verbose)

        # Load ONNX model
        ret = rknn.load_rknn(path=self.model_path)
        if ret != 0:
            print('【RKNNConfig】error :Load model failed!')
            exit(ret)
        if self.target == "RK3588":
            ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
        else:
            ret = rknn.init_runtime()
        if ret != 0:
            print('【RKNNConfig】error :Init runtime environment failed!')
            exit(ret)
        return rknn
