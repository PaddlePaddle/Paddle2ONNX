from pathlib import Path

import coremltools as ct

root = Path(__file__).parent

model_file = str(root.parent / "Models/PicoDet/picodet_l_640_coco_lcnet_nonms_no_postprocess/pico640_op11_si_sim.onnx")

model = ct.converters.onnx.convert(
    model = model_file,    
    minimum_ios_deployment_target = '13',
    image_input_names=["image"],
    preprocessing_args={
        "image_scale": 1./(0.226*255.0), 
        "red_bias": - 0.485/(0.229), # - mean / std
        "green_bias":- 0.456/(0.224),
        "blue_bias": - 0.406/(0.225)
    },
)


model.author = "Winston Fan"
model.version = "2.0"
model.short_description = "A CoreML AI Model for PicoDet from Baidu's PaddleDetection Model produced by Winston Fan @2022."
model.license = "Apache License 2.0"
model.input_description["image"] = "Input image to be detected"
saved_model_path = root.parent / "output/picodets640_nonms_nopp_sim_si"
model.save(saved_model_path)

