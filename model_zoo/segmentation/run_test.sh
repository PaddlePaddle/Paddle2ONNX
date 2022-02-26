# model=bisenet
# model=danet
# model=deeplabv3
# model=deeplabv3p
model=fcn

mkdir inference
rm -rf inference/$model
cp -r /home/x2paddle/huangshenghui/PaddleSeg/inference/$model ./inference

rm -rf cityscapes_demo.png
wget https://paddleseg.bj.bcebos.com/dygraph/demo/cityscapes_demo.png

paddle2onnx --model_dir ./inference/$model \
--model_filename model.pdmodel \
--params_filename model.pdiparams \
--save_file ./inference/$model/model.onnx \
--opset_version 11 \
--input_shape_dict="{'x':[-1,3,-1,-1]}" \
--enable_onnx_checker True

python3.7 infer.py \
    --model_path ./inference/$model/model \
    --onnx_path ./inference/$model/model.onnx \
    --image_path ./cityscapes_demo.png