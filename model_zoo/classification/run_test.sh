rm -rf inference
mkdir inference
cd inference

modol_dir=ResNet50_infer
model_url=https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/ResNet50_infer.tar

#modol_dir=PPLCNet_x1_0_infer
#model_url=https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/PPLCNet_x1_0_infer.tar

#modol_dir=MobileNetV2_infer
#model_url=https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/MobileNetV2_infer.tar

#modol_dir=MobileNetV3_small_x1_0_infer
#model_url=https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/MobileNetV3_small_x1_0_infer.tar

wget -nc $model_url && tar xf $modol_dir.tar
cd ..

paddle2onnx --model_dir=./inference/$modol_dir \
--model_filename=inference.pdmodel \
--params_filename=inference.pdiparams \
--save_file=./inference/$modol_dir/model.onnx \
--opset_version=11 \
--enable_onnx_checker=True

python3.7 infer.py \
    --model_path ./inference/$modol_dir/inference \
    --onnx_path ./inference/$modol_dir/model.onnx \
    --image_path ./images/ILSVRC2012_val_00000010.jpeg
