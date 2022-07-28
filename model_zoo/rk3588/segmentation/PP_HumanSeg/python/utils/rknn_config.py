class RKNNConfigPC:
    def __init__(self, onnx_path=None, verbose=True, do_quantization=False, dataset=None,
                 need_export=False, export_path=None):
        if onnx_path is None:
            print("【RKNNConfig】error: onnx_path is None")
            exit(0)
        else:
            print("【RKNNConfig】: rknn will load by onnx")
            self.model_path = onnx_path
        self.verbose = verbose
        self.do_quantization = do_quantization
        self.dataset = dataset
        self.need_export = need_export
        self.export_path = export_path

    def create_rknn(self):
        from rknn.api import RKNN
        # create rknn
        rknn = RKNN(verbose=False)
        # pre-process config
        rknn.config(mean_values=[[0, 0, 0]], std_values=[[1, 1, 1]],target_platform="RK3588")

        # Load ONNX model
        ret = rknn.load_onnx(model=self.model_path)
        if ret != 0:
            print('【RKNNConfig】error :Load model failed!')
            exit(ret)

        # Build model
        if self.do_quantization:
            if self.dataset is None:
                print("【RKNNConfig】error :dataset is None but need!")
                exit(0)
            ret = rknn.build(do_quantization=self.do_quantization, dataset=self.dataset)
        else:
            ret = rknn.build(do_quantization=self.do_quantization)
        if ret != 0:
            print('【RKNNConfig】error :Build model failed!')
            exit(ret)
        # rknn.accuracy_analysis(inputs=['./hrnet_demo.jpg'],target="rk3588")
        if self.need_export is not None:
            print("need ecport")
            self.rknn = rknn
            self.export_rknn()
        ret = rknn.init_runtime()
        if ret != 0:
            print('【RKNNConfig】error :Init runtime environment failed!')
            exit(ret)
        return rknn

    def export_rknn(self):
        ret = self.rknn.export_rknn(self.export_path)
        if ret != 0:
            print('RKNNConfig】error : Export rknn model failed!')
            exit(ret)


class RKNNConfigBoard:
    def __init__(self, rknn_path=None, verbose=True):
        if rknn_path is None:
            print("【RKNNConfig】error: rknn_path is None")
            exit(0)
        else:
            print("【RKNNConfig】: rknn will load by rknn")
            self.model_path = rknn_path
        self.verbose = verbose

    def create_rknn(self):
        from rknnlite.api import RKNNLite
        # create rknn
        rknn = RKNNLite(verbose=False)

        # Load ONNX model
        ret = rknn.load_rknn(path=self.model_path)
        if ret != 0:
            print('【RKNNConfig】error :Load model failed!')
            exit(ret)

        ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
        if ret != 0:
            print('【RKNNConfig】error :Init runtime environment failed!')
            exit(ret)
        return rknn