export BASEPATH=$(cd `dirname $0`; pwd)

echo ">>> Install paddle2onnx and PaddlePaddle-gpu"
python -m pip install paddlepaddle-gpu==0.0.0.post102 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html
git clone https://github.com/PaddlePaddle/Paddle2ONNX.git
cd Paddle2ONNX
python setup.py install

cd "$BASEPATH"
echo ">>> Download picodet model"
wget https://bj.bcebos.com/paddle2onnx/model_zoo/picodet_s_320_coco.tar.gz
tar xvf picodet_s_320_coco.tar.gz
echo ">>> Convert picodet model"
paddle2onnx --model_dir ./picodet_s_320_coco --model_filename model.pdmodel --params_filename model.pdiparams --save_file picodet_s_320_coco.onnx --opset_version 12 --enable_onnx_checker True --input_shape_dict "{'image': [1, 3, 320, 320]}"
echo ">>> Inference with RK PC"
python deploy.py --model_file picodet_s_320_coco.onnx --image_path images/demo.jpg --backend_type rk_pc
echo ">>> Inference finished"
