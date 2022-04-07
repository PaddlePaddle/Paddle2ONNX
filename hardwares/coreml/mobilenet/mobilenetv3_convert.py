from pathlib import Path

import coremltools as ct

root = Path(__file__).parent

def get_classes(root) -> list:
    class_id_map = {}

    img_path = root / "imagenet1k_label_list.txt"
    
    with open(str(img_path)) as file:
        while line := file.readline():
            line = line.strip()
            parts = line.partition(' ')
            values = parts[2].split(",")

            key = parts[0].strip()
            value = values[0].strip()
            if value not in class_id_map.values():
                class_id_map[key] = value
            else:
                if len(values) > 1:
                    new_value = f"{value}-{values[1].strip()}"
                else:
                    new_value = f"{key}-{value}"
                class_id_map[key] = new_value
    
    return list(class_id_map.values())

# fetch the list of classes in ImageNet1K
class_labels = get_classes(root)

model_path = str(root / "mbnv3s1_op9_sim.onnx")

# convert ONNX model to CoreML model
model = ct.converters.onnx.convert(
    mode = 'classifier',
    model = model_path,
    minimum_ios_deployment_target = '13', # set to 13 to avoid error
    class_labels = class_labels,
    preprocessing_args={
        "image_scale": 1./(0.226*255.0),
        "red_bias": - 0.485/(0.229),
        "green_bias":- 0.456/(0.224),
        "blue_bias": - 0.406/(0.225)
    },
    image_input_names= ["inputs"]
)

# add meta data to descript this model
model.author = "Winston Fan"
model.version = "1.0"
model.short_description = "A CoreML AI Model for MobileNetV3 produced by Winston Fan @2022."
model.license = "Apache License 2.0"
model.input_description["inputs"] = "Input image to be classified"
model.output_description["classLabel"] = "Image category"

saved_model_path = root / "mbnetv3_0912"
model.save(saved_model_path)

