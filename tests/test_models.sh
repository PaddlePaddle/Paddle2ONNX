export BASEPATH=$(cd `dirname $0`; pwd)
export MODELPATH="$BASEPATH/Models"
rm -r result.txt
echo "============ Model convert and diff test result =============" >> result.txt
test(){
  model_file=""
  params_file=""
  for file in $(ls $model_dir)
    do
      if [ "${file##*.}"x = "pdmodel"x ];then
        model_file=$file
        echo "find model file: $model_file"
      fi

      if [ "${file##*.}"x = "pdiparams"x ];then
        params_file=$file
        echo "find param file: $params_file"
      fi
  done
  python convert_and_check.py --model_dir=${model_dir} --paddle_model_file "$model_file" --paddle_params_file "$params_file" --input_shape=1,3,224,224
}

for dir in $(ls $MODELPATH);do
  CONVERTPATH=$MODELPATH/$dir
  echo " >>>> Model path: $CONVERTPATH"
  export model_dir=$CONVERTPATH
  test
done
