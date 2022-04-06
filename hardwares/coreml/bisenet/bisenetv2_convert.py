from pathlib import Path
import coremltools as ct

root = Path(__file__).parent

model = ct.converters.onnx.convert(
    model = "/media/winstonfan/Workspace/Work/Baidu/Hackathon2022/Models/BiSeNetV2/infermodel/BiSeNetV2_sim_of.onnx",
    minimum_ios_deployment_target = '13',
    image_input_names=["x"],
    preprocessing_args={
        "image_scale": 1./(0.5*255.0),
        "red_bias": - 0.5/(0.5),
        "green_bias":- 0.5/(0.5),
        "blue_bias": - 0.5/(0.5)
    }
)

model.author = "Winston Fan"
model.version = "1.1"
model.short_description = "A CoreML AI Model for BiSeNetV2 produced by Winston Fan @2022."
model.license = "Apache License 2.0"
model.input_description["x"] = "Input image to be segmented."
model.output_description["argmax_0.tmp_0"] = "Numpy array which contains classified data for each pixel in the input image."

saved_model_path = root.parent / "output/bisenetv2_op11_ios13"
model.save("./output/bisenetv2_op11_ios13_sim_of.mlmodel")

