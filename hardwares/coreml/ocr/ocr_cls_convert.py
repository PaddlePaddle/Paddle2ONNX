from pathlib import Path
import coremltools as ct
root = Path(__file__).parent

model = ct.converters.onnx.convert(
    model = "cls_sim_fo.onnx",
    minimum_ios_deployment_target = '13',
    image_input_names= ["x"]
)

model.author = "Winston Fan"
model.version = "1.0"
model.short_description = "A CoreML AI Model for PPOCRV2xx_cls_infer produced by Winston Fan @2022. This model is only used for Text Direction Classification. The result will be 0 or 180 degree."
model.license = "Apache License 2.0"
model.input_description["x"] = "Input image to be detected"
model.output_description["save_infer_model/scale_0.tmp_1"] = "This result shows the degree of text angle, e.g. result: ('0', 0.9998784)"

# model.save("./output/mbnetv3.mlpackage") # only available from coremltools V5
saved_model_path = root.parent / "output/ocr_cls"
model.save(saved_model_path)

