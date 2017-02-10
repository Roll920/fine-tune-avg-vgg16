#!/bin/bash
python generate_prototxt.py
TOOLS=/home/luojh2/Software/caffe-master/build/tools
model_path=/data/luojh/net/caffe/fc_conv_VGG_ILSVRC_16_layers.caffemodel
gpu=6,7

log_name="AVG_VGG16.log"
LOG=avg_vgg/logs/${log_name}
if [ ! -d "avg_vgg/logs" ]; then
   mkdir avg_vgg/logs
fi
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

$TOOLS/caffe train --solver=avg_vgg/solver.prototxt -weights $model_path -gpu $gpu

cd avg_vgg/logs/
../../parse_log/parse_log.sh "$log_name"
cd ../..
