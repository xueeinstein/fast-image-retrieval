#!/bin/bash

# check caffe
if [[ $CAFFEROOT == '' ]]; then
  echo "\$CAFFEROOT not found, please define it at first."
  echo "export CAFFEROOT=/path/to/your/caffe"
  exit
fi

# init env
TOOLS=$CAFFEROOT/build/tools
BASEDIR=$(dirname "$0")
PROJROOT=$BASEDIR/../../
CURRDIR=$(pwd)

# check dataset
if [ ! -d $BASEDIR/facescrub_train_lmdb ] || [ ! -d $BASEDIR/facescrub_test_lmdb ] ; then
  echo "The facescrub dataset folder cannot found"
  echo "Please execute convert_facescrub_data.py at first"
  exit
fi

# compute mean
$TOOLS/compute_image_mean $BASEDIR/facescrub_train_lmdb $BASEDIR/facescrub_mean.binaryproto
python $PROJROOT/tools/convert_protomean.py $BASEDIR/facescrub_mean.binaryproto $BASEDIR/facescrub_mean.npy

cd $BASEDIR

# pretrain
$TOOLS/caffe train --solver=solver.prototxt 2>&1 | tee pretrain.log

# train net with latent layer
$TOOLS/caffe train \
  --solver=solver_with_latent_layer.prototxt \
  --weights=facescrub_model_iter_10000.caffemodel 2>&1 | tee train.log

cd $CURRDIR
