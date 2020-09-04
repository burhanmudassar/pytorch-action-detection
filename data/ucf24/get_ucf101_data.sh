#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $DIR

DATASET_NAME="UCF101"

declare -a RGB_FLOW_NAMES=("Frames" "OF")
RGBFLOW="${RGB_FLOW_NAMES[$1]}"

declare -a CHECKSUM_RGB_Flow=("aa6c870aded72ae4820944ff62098b8e" "cace8ea49d548ca863bc6bd0f1e07ee0")
CHECKSUM="${CHECKSUM_RGB_Flow[$1]}"

FILE=$DATASET_NAME"-"$RGBFLOW".tar.gz"

if [ -f $FILE ]; then
  echo "File already exists. Checking md5..."
  os=`uname -s`
  if [ "$os" = "Linux" ]; then
    checksum=`md5sum $FILE | awk '{ print $1 }'`
  elif [ "$os" = "Darwin" ]; then
    checksum=`cat $FILE | md5`
  fi
  if [ "$checksum" = "$CHECKSUM" ]; then
    echo "Dataset "$DATASET_NAME" checksum is correct. No need to download."
    exit 0
  else
    echo "Dataset "$DATASET_NAME" checksum is incorrect. Need to download again."
  fi
fi

echo "Downloading "$FILE" used in ACT-detector ..."

wget http://pascal.inrialpes.fr/data2/act-detector/downloads/datasets/$FILE

echo "Unzipping..."

tar zxvf $FILE  

echo "Done. Please run this command again to verify that checksum = $CHECKSUM."