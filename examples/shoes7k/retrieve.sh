#!/bin/bash

# init env
BASEDIR=$(dirname "$0")
PROJROOT=$BASEDIR/../..
MODEL_FILE="$BASEDIR/shoes7k_model_with_latent_layer_iter_10000.caffemodel"
DEPLOY_FILE="$BASEDIR/deploy_with_latent_layer.prototxt"
MEAN_FILE="$BASEDIR/shoes7k_mean.npy"
IMAGE_NPY="$BASEDIR/image_files.npy"
FC7_NPY="$BASEDIR/fc7_features.npy"
LATENT_NPY="$BASEDIR/latent_features.npy"
TARGET="$1"

# check model
if [ ! -e $MODEL_FILE ] || [ ! -e $DEPLOY_FILE ] || [ ! -e $MEAN_FILE ]; then
  echo "Please train the model at first"
  echo "./train.sh"
  exit
fi

# parse parsemeters
if [[ $TARGET == "-h" ]] || [[ $TARGET == "--help" ]]; then
  echo "Usage: ./retrieve.sh image_to_retrieve.jpg"
  exit
fi

if [ -e $IMAGE_NPY ] && [ -e $FC7_NPY ] && [ -e $LATENT_NPY ]; then
  python $PROJROOT/retrieve.py $MODEL_FILE $DEPLOY_FILE $MEAN_FILE $TARGET
else
  echo "generate feature matrix..."
  python $BASEDIR/generate_feature_mat.py $MODEL_FILE $DEPLOY_FILE $MEAN_FILE
  python $PROJROOT/retrieve.py $MODEL_FILE $DEPLOY_FILE $MEAN_FILE $TARGET
fi

