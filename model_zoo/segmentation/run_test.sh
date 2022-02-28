PaddleSegPATH=/home/x2paddle/huangshenghui/PaddleSeg
Paddle2ONNXPATH=/home/x2paddle/huangshenghui/Paddle2ONNX
cd $PaddleSegPATH
mkdir -p inference/

model=bisenet
config_file=bisenet_cityscapes_1024x1024_160k.yml
model_url=https://bj.bcebos.com/paddleseg/dygraph/cityscapes/bisenet_cityscapes_1024x1024_160k/model.pdparams

# model=danet
# config_file=danet_resnet50_os8_cityscapes_1024x512_80k.yml
# model_url=https://bj.bcebos.com/paddleseg/dygraph/cityscapes/danet_resnet50_os8_cityscapes_1024x512_80k/model.pdparams

# model=deeplabv3
# config_file=deeplabv3_resnet50_os8_cityscapes_1024x512_80k.yml
# model_url=https://bj.bcebos.com/paddleseg/dygraph/cityscapes/deeplabv3_resnet50_os8_cityscapes_1024x512_80k/model.pdparams

# model=deeplabv3p
# config_file=deeplabv3p_resnet50_os8_cityscapes_1024x512_80k.yml
# model_url=https://bj.bcebos.com/paddleseg/dygraph/cityscapes/deeplabv3p_resnet50_os8_cityscapes_1024x512_80k/model.pdparams

# model=fcn
# config_file=fcn_hrnetw18_cityscapes_1024x512_80k.yml
# model_url=https://bj.bcebos.com/paddleseg/dygraph/cityscapes/fcn_hrnetw18_cityscapes_1024x512_80k/model.pdparams

rm -rf inference/$model
mkdir -p inference/$model
cd inference/$model
wget $model_url
cd ../..

python3.7 export.py \
       --config configs/$model/$config_file \
       --model_path inference/$model/model.pdparams\
       --save_dir inference/$model

rm -rf inference/$model/model.pdparams

cd $Paddle2ONNXPATH/model_zoo/segmentation
mkdir inference
rm -rf inference/$model
cp -r $PaddleSegPATH/inference/$model ./inference

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