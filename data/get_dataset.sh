#!/bin/bash

if [ "$1" = "ucfsports" ]; then
    sh ./data/ucfsports/get_ucfsports_data.sh 0
    sh ./data/ucfsports/get_ucfsports_data.sh 1
    data_dir="./data/ucfsports"
    cache_link=https://drive.google.com/uc?id=1xK-0ArCuMrghOYiEeNTq1-M2Jk1_UARl
    elif [ "$1" = "jhmdb" ]; then
    sh ./data/jhmdb/get_jhmdb_data.sh 0
    sh ./data/jhmdb/get_jhmdb_data.sh 1
    data_dir="./data/jhmdb"
    cache_link=https://drive.google.com/uc?1aLkqIyq2BULQWaQGFyir3LJpIYCNaqv-
    elif [ "$1" = "ucf24" ]; then
    sh ./data/jhmdb/get_ucf101_data.sh 0
    sh ./data/jhmdb/get_ucf101_data.sh 1
    data_dir="./data/ucf24"
    cache_link=https://drive.google.com/uc?1P7ww6KvNeIlgM8ymU5Qz1PQ20G-eEruX
    else
    echo "Dataset not Recognized"
    exit 1
fi

pushd $PWD
cd $data_dir
ln -s Frames rgb-images
ln -s FlowBrox04 brox-images
gdown --no-cookies $cache_link
unzip "$1_cache.zip"
rm "$1_cache.zip"
popd

