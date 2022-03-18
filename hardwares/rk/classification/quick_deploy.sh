export BASEPATH=$(cd `dirname $0`; pwd)

echo ">>> Install paddle2onnx and PaddlePaddle-gpu"
python -m pip install paddlepaddle-gpu==0.0.0.post102 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html
git clone https://github.com/PaddlePaddle/Paddle2ONNX.git
cd Paddle2ONNX
python setup.py install

cd "$BASEPATH"
echo ">>> Download mobilenetv3 model"
wget https://bj.bcebos.com/paddle2onnx/model_zoo/mobilenetv3.tar.gz
tar xvf mobilenetv3.tar.gz
echo ">>> Convert mobilenetv3 model"
paddle2onnx --model_dir ./mobilenetv3 --model_filename inference.pdmodel --params_filename inference.pdiparams --save_file mobilenetv3.onnx --opset_version 12 --enable_onnx_checker True  --input_shape_dict "{'inputs': [1, 3, 224, 224]}"
echo ">>> Inference with RK PC"
python deploy.py --model_file mobilenetv3.onnx --image_path images/ILSVRC2012_val_00000010.jpeg --backend_type rk_pc
echo ">>> Inference finished"