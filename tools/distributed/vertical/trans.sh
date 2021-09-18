#!/bin/sh
for i in `ls *.onnx`; do
    ./onnx2ncnn $i
done
