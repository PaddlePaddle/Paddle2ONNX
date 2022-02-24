
# wget https://paddledet.bj.bcebos.com/models/yolov3_mobilenet_v1_270e_coco.pdparams
# https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/yolov3/yolov3_mobilenet_v1_270e_coco.yml

#https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/yolov3/yolov3_darknet53_270e_coco.yml


#rm -rf inference
#mkdir inference
#cd inference
#
#wget https://paddledet.bj.bcebos.com/models/yolov3_darknet53_270e_coco.pdparams
#cd ..
#
#python3.7 tools/export_model.py -c configs/yolov3/yolov3_darknet53_270e_coco.yml --output_dir=./inference \
# -o weights=inference/yolov3_darknet53_270e_coco.pdparams use_gpu=false

python3.7 infer.py --model_dir=./inference/yolov3_darknet53_270e_coco --image_file=./images/000000014439.jpg --device=cpu
python3.7 infer.py --model_dir=./inference/yolov3_darknet53_270e_coco --image_file=./images/000000087038.jpg --device=cpu
python3.7 infer.py --model_dir=./inference/yolov3_darknet53_270e_coco --image_file=./images/000000570688.jpg --device=cpu
python3.7 infer.py --model_dir=./inference/yolov3_darknet53_270e_coco --image_file=./images/hrnet_demo.jpg --device=cpu
python3.7 infer.py --model_dir=./inference/yolov3_darknet53_270e_coco --image_file=./images/orange_71.jpg --device=cpu
