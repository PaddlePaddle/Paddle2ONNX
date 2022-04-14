from pathlib import Path
import coremltools as ct
root = Path(__file__).parent

# download the onnx model to the same folder before running the following code
model = ct.converters.onnx.convert(
    model = "det_sim_fo.onnx",
    minimum_ios_deployment_target = '13',
    preprocessing_args={
        "image_scale": 1./(0.226*255.0),
        "red_bias": - 0.485/(0.229),
        "green_bias":- 0.456/(0.224),
        "blue_bias": - 0.406/(0.225)
    },
    image_input_names= ["x"]
)

model.author = "Winston Fan"
model.version = "1.0"
model.short_description = "A CoreML AI Model for PPOCRV2xx_det_infer produced by Winston Fan @2022."
model.license = "Apache License 2.0"
model.input_description["x"] = "Input image to be detected"

# model.save("./output/mbnetv3.mlpackage") # only available from coremltools V5
saved_model_path = root.parent / "output/ocr_det"
model.save(saved_model_path)

