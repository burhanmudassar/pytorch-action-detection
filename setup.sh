#!/bin/bash
pip3 install -r requirements.txt

pushd $PWD
cd lib
make
popd
